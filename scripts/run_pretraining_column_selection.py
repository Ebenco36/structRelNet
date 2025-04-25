#!/usr/bin/env python
"""
scripts/run_pretraining_column_selection.py

Column‑Selection pretraining for StructRelNet:
- Iterates one table at a time (no batching)
- Pools cell embeddings into column embeddings
- Trains col_head via BCEWithLogitsLoss
- Saves only col_head’s weights
"""
import os
import sys
import argparse
import torch
import torch.nn as nn

# Ensure project root on PYTHONPATH
sys.path.append(os.getcwd())

from table_parser.table_factory    import TableLoaderFactory
from graph_builder.build_graph     import StructRelGraphBuilder
from embedding.value_encoder       import ValueEncoder
from embedding.type_encoder        import TypeEncoder
from embedding.position_encoder    import PositionEncoder
from embedding.schema_encoder      import SchemaEncoder
from embedding.combine_embeddings  import CombineEmbeddings
from model.structrelnet            import StructRelNet

def build_graph_data(path, value_enc, type_enc, pos_enc, schema_enc, combiner, device):
    # 1) Load & build
    table = TableLoaderFactory.load(path)
    StructRelGraphBuilder().build(table)

    # 2) Flatten cells
    n_rows, n_cols = len(table.rows), len(table.header)
    values, types, rows, cols = [], [], [], []
    for r_idx, row in enumerate(table.rows):
        for c_idx, cv in enumerate(row):
            values.append(cv)
            types.append(table.types[c_idx] if table.types else 'unknown')
            rows.append(r_idx)
            cols.append(c_idx)

    # 3) Encode modalities
    schema_str = ' '.join(table.header)
    schemas    = [schema_str] * len(values)
    val_emb    = value_enc(values).to(device)
    type_emb   = type_enc(types).to(device)
    pos_emb    = pos_enc(rows, cols).to(device)
    schema_emb = schema_enc(schemas).to(device)

    # 4) Fuse & detach
    fused = combiner(val_emb, type_emb, pos_emb, schema_emb).detach().cpu()

    # 5) Build same‑row & same‑column edges + self‑loops
    src, dst = [], []
    # same‑row
    for r in range(n_rows):
        for i in range(n_cols):
            for j in range(n_cols):
                if i != j:
                    idx1 = r * n_cols + i
                    idx2 = r * n_cols + j
                    src.append(idx1); dst.append(idx2)
    # same‑column
    for c in range(n_cols):
        for i in range(n_rows):
            for j in range(n_rows):
                if i != j:
                    idx1 = i * n_cols + c
                    idx2 = j * n_cols + c
                    src.append(idx1); dst.append(idx2)
    # self‑loops
    for idx in range(n_rows * n_cols):
        src.append(idx); dst.append(idx)

    edge_index = torch.stack([
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long)
    ], dim=0)

    return fused, edge_index, n_rows, n_cols

def main():
    parser = argparse.ArgumentParser("Run StructRelNet Column‑Selection pretraining")
    parser.add_argument('--data_dir',     type=str, default='data/raw')
    parser.add_argument('--value_model',  type=str, default='bert-base-uncased')
    parser.add_argument('--schema_model', type=str, default='bert-base-uncased')
    parser.add_argument('--epochs',       type=int, default=10)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--gnn_hidden',   type=int, default=128)
    parser.add_argument('--gnn_out',      type=int, default=128)
    parser.add_argument('--transformer_heads',  type=int, default=4)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--num_agg_classes', type=int, default=4)
    args = parser.parse_args()

    # Reproducibility & device
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate encoders & combiner
    value_enc  = ValueEncoder(model_name=args.value_model, device=device)
    type_enc   = TypeEncoder(type_list=['cell','header','column','row'], embedding_dim=32)
    pos_enc    = PositionEncoder(dim=32)
    schema_enc = SchemaEncoder(model_name=args.schema_model, device=device)
    combiner   = CombineEmbeddings(fused_dim=args.gnn_hidden)

    # Prepare graph data for each table
    files = sorted(f for f in os.listdir(args.data_dir)
                   if os.path.isfile(os.path.join(args.data_dir, f)))
    samples = [
        build_graph_data(
            os.path.join(args.data_dir, fn),
            value_enc, type_enc, pos_enc, schema_enc,
            combiner, device
        )
        for fn in files
    ]

    # Load pretrained StructRelNet (allow missing new heads)
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
        num_agg_classes    = args.num_agg_classes
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)

    # Optimizer & loss on col_head
    optimizer = torch.optim.Adam(model.col_head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    print("Starting Column‑Selection pretraining…")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for feats, eidx, n_rows, n_cols in samples:
            feats = feats.to(device)   # [n_rows * n_cols, input_dim]
            eidx  = eidx.to(device)

            # 1) cell embeddings
            cell_emb = model.gnn(feats, eidx)  # [n_cells, gnn_out]

            # 2) reshape & pool per column → [n_cols, gnn_out]
            col_emb = cell_emb.view(n_rows, n_cols, -1).mean(dim=0)

            # 3) dummy labels (one per column)
            col_labels = torch.zeros(n_cols, dtype=torch.float32, device=device)

            # 4) predict & compute loss
            col_logits = model.col_head(col_emb).squeeze(-1)  # [n_cols]
            loss = criterion(col_logits, col_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(samples)
        print(f"Epoch {epoch}/{args.epochs} — Avg Col‑Sel loss: {avg:.4f}")

    # Save only the col_head weights
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.col_head.state_dict(), 'checkpoints/col_selector.pt')
    print("✅ Saved column selector to checkpoints/col_selector.pt")

if __name__ == '__main__':
    main()
