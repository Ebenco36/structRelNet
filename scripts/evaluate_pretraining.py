#!/usr/bin/env python
"""
scripts/evaluate_pretraining.py

Evaluate pre‑training heads (MCM, Row, Column, Aggregation) on held‑out tables,
using real labels from a metadata JSON.
"""
import os
import sys
import argparse
import json

import torch
import torch.nn as nn
import numpy as np

# Ensure project root is on PYTHONPATH so imports work
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
    """
    Load a table and build its fused features + edge_index.
    Returns: fused [n_cells, dim], edge_index [2, E], n_rows, n_cols
    """
    table = TableLoaderFactory.load(path)
    StructRelGraphBuilder().build(table)

    n_rows, n_cols = len(table.rows), len(table.header)
    vals, types, rows, cols = [], [], [], []

    for r in range(n_rows):
        for c in range(n_cols):
            vals.append(table.rows[r][c])
            types.append(table.types[c] if table.types else 'unknown')
            rows.append(r)
            cols.append(c)

    schema_str = ' '.join(table.header)
    schemas    = [schema_str] * len(vals)

    val_emb    = value_enc(vals).to(device)
    type_emb   = type_enc(types).to(device)
    pos_emb    = pos_enc(rows, cols).to(device)
    schema_emb = schema_enc(schemas).to(device)

    fused = combiner(val_emb, type_emb, pos_emb, schema_emb).detach().cpu()

    # build same-row, same-col, self-loop edges
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

    return fused, edge_index, n_rows, n_cols

def parse_args():
    p = argparse.ArgumentParser("Evaluate StructRelNet pretraining")
    p.add_argument(
        '--val_dir',    type=str, required=True,
        help="Directory of held‑out table files (csv/tsv/...)" 
    )
    p.add_argument(
        '--metadata',   type=str, required=True,
        help="Path to train_metadata.json with real labels"
    )
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.isdir(args.val_dir):
        print(f"Error: validation directory '{args.val_dir}' not found.")
        sys.exit(1)
    if not os.path.isfile(args.metadata):
        print(f"Error: metadata file '{args.metadata}' not found.")
        sys.exit(1)

    # Load metadata JSON
    with open(args.metadata) as f:
        meta_list = json.load(f)
    # build lookup: table_id (filename) -> entry
    meta = { entry['table_id']: entry for entry in meta_list }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate encoders + combiner
    value_enc  = ValueEncoder(device=device)
    type_enc   = TypeEncoder(type_list=['cell','header','column','row'], embedding_dim=32)
    pos_enc    = PositionEncoder(dim=32)
    schema_enc = SchemaEncoder(device=device)
    combiner   = CombineEmbeddings(fused_dim=128)

    # Load pretrained encoder checkpoint
    pretrain_ckpt = 'checkpoints/pretrained_model.pt'
    if not os.path.isfile(pretrain_ckpt):
        raise FileNotFoundError(f"Checkpoint '{pretrain_ckpt}' not found.")
    state = torch.load(pretrain_ckpt, map_location=device)

    # Infer original agg_head size from checkpoint
    if 'agg_head.bias' not in state:
        raise KeyError("Key 'agg_head.bias' not in pretrained_model.pt")
    n_agg_classes = state['agg_head.bias'].shape[0]

    # Build model with matching agg_head dimension
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
        num_agg_classes    = n_agg_classes
    ).to(device)

    # Load state dict (will ignore missing heads if any)
    model.load_state_dict(state, strict=False)

    # Now load each head’s checkpoint
    def load_head(name, path):
        if os.path.isfile(path):
            getattr(model, name).load_state_dict(
                torch.load(path, map_location=device),
                strict=False
            )
        else:
            print(f"Warning: head checkpoint '{path}' not found, skipping {name}")

    load_head('row_head', 'checkpoints/row_selector.pt')
    load_head('col_head', 'checkpoints/col_selector.pt')
    load_head('agg_head', 'checkpoints/agg_head.pt')

    model.eval()

    # Loss functions
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()

    mcm_losses, row_accs, col_accs, agg_accs = [], [], [], []

    # Iterate over validation tables
    for fn in sorted(os.listdir(args.val_dir)):
        # skip non-table files
        if not fn.lower().endswith(('.csv','.tsv','.xls','.xlsx','.html','.htm')):
            continue
        path = os.path.join(args.val_dir, fn)
        fused, eidx, n_rows, n_cols = build_graph(
            path, value_enc, type_enc, pos_enc, schema_enc, combiner, device
        )
        fused, eidx = fused.to(device), eidx.to(device)

        # Lookup labels by exact filename
        if fn not in meta:
            print(f"Warning: no metadata for '{fn}', skipping.")
            continue
        entry   = meta[fn]
        row_lbl = torch.tensor(entry['row_labels'], dtype=torch.float32)
        col_lbl = torch.tensor(entry['col_labels'], dtype=torch.float32)
        agg_lbl = torch.tensor(entry['agg_label'],  dtype=torch.long)

        # 1) MCM reconstruction
        with torch.no_grad():
            out_feats = model.gnn(fused, eidx)
        mcm_losses.append(mse(out_feats, fused).item())

        # 2) Row selection
        row_emb    = out_feats.view(n_rows, n_cols, -1).mean(dim=1)
        row_logits = model.row_head(row_emb).squeeze(-1)
        row_pred   = (row_logits.sigmoid() > 0.5).float().cpu()
        row_accs.append((row_pred == row_lbl).float().mean().item())

        # 3) Column selection
        col_emb    = out_feats.view(n_rows, n_cols, -1).mean(dim=0)
        col_logits = model.col_head(col_emb).squeeze(-1)
        col_pred   = (col_logits.sigmoid() > 0.5).float().cpu()
        col_accs.append((col_pred == col_lbl).float().mean().item())

        # 4) Aggregation
        with torch.no_grad():
            _, _, agg_logits = model(fused, eidx)
        agg_pred = agg_logits.argmax(dim=-1).cpu()
        agg_accs.append((agg_pred == agg_lbl).float().item())

    # Report
    print("\n=== Pretraining Evaluation ===")
    print(f"MCM   MSE:    {np.mean(mcm_losses):.4f}")
    print(f"Row   Acc:    {np.mean(row_accs)*100:.2f}%")
    print(f"Col   Acc:    {np.mean(col_accs)*100:.2f}%")
    print(f"Agg   Acc:    {np.mean(agg_accs)*100:.2f}%\n")

if __name__ == "__main__":
    main()
