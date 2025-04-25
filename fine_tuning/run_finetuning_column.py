#!/usr/bin/env python
"""
scripts/run_finetuning_column.py

Fine‑tune the StructRelNet column‐selection head on your labeled tables.
"""
import os
import sys
import json
import argparse

import torch
import torch.nn as nn

# Ensure project root is on PYTHONPATH
sys.path.append(os.getcwd())

from table_parser.table_factory    import TableLoaderFactory
from graph_builder.build_graph     import StructRelGraphBuilder
from embedding.value_encoder       import ValueEncoder
from embedding.type_encoder        import TypeEncoder
from embedding.position_encoder    import PositionEncoder
from embedding.schema_encoder      import SchemaEncoder
from embedding.combine_embeddings  import CombineEmbeddings
from model.structrelnet            import StructRelNet

def build_graph(path, value_enc, type_enc, pos_enc, schema_enc, combiner, device):
    """Load table → fused features, edge_index, n_rows, n_cols."""
    table = TableLoaderFactory.load(path)
    StructRelGraphBuilder().build(table)
    header, rows = table.header, table.rows
    n_rows, n_cols = len(rows), len(header)
    vals, types, rs, cs = [], [], [], []
    for r in range(n_rows):
        for c in range(n_cols):
            vals.append(rows[r][c])
            types.append(table.types[c] if table.types else 'unknown')
            rs.append(r); cs.append(c)
    schema_str = ' '.join(header)
    schemas    = [schema_str]*len(vals)

    # Encode
    val_emb    = value_enc(vals).to(device)
    type_emb   = type_enc(types).to(device)
    pos_emb    = pos_enc(rs, cs).to(device)
    schema_emb = schema_enc(schemas).to(device)
    fused      = combiner(val_emb, type_emb, pos_emb, schema_emb).detach().cpu()

    # Build edges
    src, dst = [], []
    for r in range(n_rows):
        for i in range(n_cols):
            for j in range(n_cols):
                if i!=j:
                    src.append(r*n_cols+i); dst.append(r*n_cols+j)
    for c in range(n_cols):
        for i in range(n_rows):
            for j in range(n_rows):
                if i!=j:
                    src.append(i*n_cols+c); dst.append(j*n_cols+c)
    for idx in range(n_rows*n_cols):
        src.append(idx); dst.append(idx)

    edge_index = torch.stack([
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long)
    ], dim=0)

    return fused, edge_index, n_rows, n_cols

def parse_args():
    p = argparse.ArgumentParser("Fine‑tune column head")
    p.add_argument('--data_dir',  type=str, required=True,
                   help="Directory of raw tables (e.g. data/raw)")
    p.add_argument('--metadata',  type=str, required=True,
                   help="Path to train_metadata.json")
    p.add_argument('--epochs',    type=int, default=5)
    p.add_argument('--lr',        type=float, default=5e-4)
    p.add_argument('--seed',      type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    # Load metadata
    with open(args.metadata) as f:
        meta_list = json.load(f)
    meta = { e['table_id']: e for e in meta_list }

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build encoder + combiner
    value_enc  = ValueEncoder(device=device)
    type_enc   = TypeEncoder(type_list=['cell','header','column','row'], embedding_dim=32)
    pos_enc    = PositionEncoder(dim=32)
    schema_enc = SchemaEncoder(device=device)
    combiner   = CombineEmbeddings(fused_dim=128)

    # Load pre‑trained StructRelNet and col_head
    pretrain_ckpt = 'checkpoints/pretrained_model.pt'
    state = torch.load(pretrain_ckpt, map_location=device)
    # infer agg classes to match checkpoint
    n_agg = state['agg_head.bias'].shape[0]
    model = StructRelNet(
        input_dim          = combiner.fused_dim,
        gnn_hidden         = 128,
        gnn_out            = 128,
        transformer_heads  = 4,
        transformer_layers = 2,
        gnn_type           = 'gcn',
        gnn_layers         = 2,
        dropout            = 0.1,
        use_residual       = True,
        num_agg_classes    = n_agg
    ).to(device)
    model.load_state_dict(state, strict=False)
    # load col_head
    col_ckpt = 'checkpoints/col_selector.pt'
    if os.path.isfile(col_ckpt):
        model.col_head.load_state_dict(torch.load(col_ckpt, map_location=device),
                                      strict=False)
    model.train()

    # Prepare training samples
    samples = []
    for fname in sorted(os.listdir(args.data_dir)):
        if not fname.lower().endswith(('.csv','.tsv','.xls','.xlsx','.html','.htm')):
            continue
        if fname not in meta:
            print(f"Warning: no metadata for '{fname}', skipping.")
            continue
        path = os.path.join(args.data_dir, fname)
        fused, eidx, n_rows, n_cols = build_graph(
            path, value_enc, type_enc, pos_enc, schema_enc, combiner, device
        )
        # true column labels
        col_lbl = torch.tensor(meta[fname]['col_labels'], dtype=torch.float32)
        samples.append((fused, eidx, n_rows, n_cols, col_lbl))

    optimizer = torch.optim.Adam(model.col_head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Fine‑tuning column head on {len(samples)} tables…")
    for epoch in range(1, args.epochs+1):
        total, correct = 0, 0
        epoch_loss = 0.0

        for fused, eidx, n_rows, n_cols, col_lbl in samples:
            fused, eidx = fused.to(device), eidx.to(device)
            col_lbl = col_lbl.to(device)

            optimizer.zero_grad()
            # forward pass
            cell_emb = model.gnn(fused, eidx)           # [n_cells, hidden]
            col_emb  = cell_emb.view(n_rows, n_cols, -1).mean(dim=0)  # [n_cols, hidden]
            logits   = model.col_head(col_emb).squeeze(-1)           # [n_cols]
            loss     = criterion(logits, col_lbl)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            preds = (logits.sigmoid() > 0.5).float().cpu()
            correct += (preds == col_lbl.cpu()).sum().item()
            total   += col_lbl.numel()

        acc = correct/total * 100
        print(f"Epoch {epoch}/{args.epochs} — Loss: {epoch_loss/len(samples):.4f} — Acc: {acc:.2f}%")

    # Save fine‑tuned col_head
    os.makedirs('checkpoints', exist_ok=True)
    outp = 'checkpoints/col_finetuned.pt'
    torch.save(model.col_head.state_dict(), outp)
    print(f"✅ Saved fine‑tuned col_head to {outp}")

if __name__ == '__main__':
    main()
