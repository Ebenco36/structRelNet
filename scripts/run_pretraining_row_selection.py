#!/usr/bin/env python
"""
scripts/run_pretraining_row_selection.py

Row‑Selection pretraining for StructRelNet (manual loop, correct row granularity):
- Pools cell embeddings into row embeddings before classification.
"""
import os
import sys
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

def build_graph_dict(path, value_enc, type_enc, pos_enc, schema_enc, combiner, device):
    table = TableLoaderFactory.load(path)
    StructRelGraphBuilder().build(table)

    # flatten cells
    n_rows, n_cols = len(table.rows), len(table.header)
    vals, types, rows, cols = [], [], [], []
    for r, row in enumerate(table.rows):
        for c, cv in enumerate(row):
            vals.append(cv)
            types.append(table.types[c] if table.types else 'unknown')
            rows.append(r)
            cols.append(c)

    schema_str = ' '.join(table.header)
    schemas    = [schema_str] * len(vals)

    # encode
    val_emb    = value_enc(vals).to(device)
    type_emb   = type_enc(types).to(device)
    pos_emb    = pos_enc(rows, cols).to(device)
    schema_emb = schema_enc(schemas).to(device)

    fused = combiner(val_emb, type_emb, pos_emb, schema_emb).detach().cpu()

    # edges + self loops
    src, dst = [], []
    for r in range(n_rows):
        for i in range(n_cols):
            for j in range(n_cols):
                if i != j:
                    src.append(r*n_cols + i); dst.append(r*n_cols + j)
    for c in range(n_cols):
        for i in range(n_rows):
            for j in range(n_rows):
                if i != j:
                    src.append(i*n_cols + c); dst.append(j*n_cols + c)
    for idx in range(n_rows * n_cols):
        src.append(idx); dst.append(idx)

    edge_index = torch.stack([
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long)
    ], dim=0)

    # dummy row labels, one per row
    row_labels = torch.zeros(n_rows, dtype=torch.float32)

    return fused, edge_index, row_labels

def main():
    p = argparse.ArgumentParser("Row‑Selection pretrain")
    p.add_argument('--data_dir',     type=str, default='data/raw')
    p.add_argument('--value_model',  type=str, default='bert-base-uncased')
    p.add_argument('--schema_model', type=str, default='bert-base-uncased')
    p.add_argument('--epochs',       type=int, default=10)
    p.add_argument('--lr',           type=float, default=5e-4)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--gnn_hidden',   type=int, default=128)
    p.add_argument('--gnn_out',      type=int, default=128)
    p.add_argument('--transformer_heads',  type=int, default=4)
    p.add_argument('--transformer_layers', type=int, default=2)
    p.add_argument('--dropout',      type=float, default=0.1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate encoders + combiner
    value_enc  = ValueEncoder(model_name=args.value_model,  device=device)
    type_enc   = TypeEncoder(type_list=['cell','header','column','row'], embedding_dim=32)
    pos_enc    = PositionEncoder(dim=32)
    schema_enc = SchemaEncoder(model_name=args.schema_model, device=device)
    combiner   = CombineEmbeddings(fused_dim=args.gnn_hidden)

    # prepare dataset
    files = sorted(os.listdir(args.data_dir))
    samples = [
        build_graph_dict(
            os.path.join(args.data_dir, fn),
            value_enc, type_enc, pos_enc, schema_enc,
            combiner, device
        )
        for fn in files if os.path.isfile(os.path.join(args.data_dir, fn))
    ]

    # load pretrained encoder
    ckpt = 'checkpoints/pretrained_model.pt'
    if not os.path.exists(ckpt):
        raise FileNotFoundError("Run MCM pretraining first.")
    model = StructRelNet(
        input_dim          = combiner.fused_dim,
        gnn_hidden         = args.gnn_hidden,
        gnn_out            = args.gnn_out,
        transformer_heads  = args.transformer_heads,
        transformer_layers = args.transformer_layers,
        gnn_type           = 'gcn',
        gnn_layers         = 2,
        dropout            = args.dropout,
        use_residual       = True,
        num_agg_classes    = 4
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    optimizer = torch.optim.Adam(model.row_head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting Row‑Selection pretraining…")
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0

        for feats, eidx, row_labels in samples:
            feats = feats.to(device)
            eidx  = eidx.to(device)
            row_labels = row_labels.to(device)

            # get cell embeddings
            cell_emb = model.gnn(feats, eidx)             # [n_cells, hidden]
            # reshape → [n_rows, n_cols, hidden]
            n_cells = cell_emb.size(0)
            n_rows   = row_labels.size(0)
            n_cols   = n_cells // n_rows
            row_emb  = cell_emb.view(n_rows, n_cols, -1).mean(dim=1)  # [n_rows, hidden]

            row_logits = model.row_head(row_emb).squeeze(-1)         # [n_rows]

            loss = criterion(row_logits, row_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(samples)
        print(f"Epoch {epoch}/{args.epochs} — Avg loss: {avg:.4f}")

    # save only row_head weights
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.row_head.state_dict(), 'checkpoints/row_selector.pt')
    print("✅ Saved row selector to checkpoints/row_selector.pt")

if __name__ == '__main__':
    main()


# python scripts/run_pretraining_row_selection.py --data_dir data/raw --epochs 5 --lr 5e-4