#!/usr/bin/env python
"""
scripts/run_pretraining.py

Masked Cell Modeling (MCM) pretraining for StructRelNet:
1. Load tables via TableLoaderFactory
2. Optionally build full StructRel graph
3. Encode & fuse cell features (detached)
4. Build same‑row/column + self‑loops edge_index
5. Train GNN on MCM with a custom loop (robust sample parsing)
6. Save pretrained weights
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
from pretraining.mcm_pretraining   import MaskedCellModelingDataset
from model.structrelnet            import StructRelNet

def build_graph_data(path, value_enc, type_enc, pos_enc, schema_enc, combiner, device):
    # Load and graph‐build
    table = TableLoaderFactory.load(path)
    StructRelGraphBuilder().build(table)

    # Flatten cells
    n_rows, n_cols = len(table.rows), len(table.header)
    values, types, rows, cols = [], [], [], []
    for r_idx, row in enumerate(table.rows):
        for c_idx, cv in enumerate(row):
            values.append(cv)
            types.append(table.types[c_idx] if table.types else 'unknown')
            rows.append(r_idx)
            cols.append(c_idx)
    schema_str = ' '.join(table.header)
    schemas    = [schema_str] * len(values)

    # Encode modalities
    val_emb    = value_enc(values).to(device)
    type_emb   = type_enc(types).to(device)
    pos_emb    = pos_enc(rows, cols).to(device)
    schema_emb = schema_enc(schemas).to(device)

    # Fuse and detach (so combiner isn’t in the MCM graph)
    fused = combiner(val_emb, type_emb, pos_emb, schema_emb).detach().cpu()

    # Build same‑row + same‑column edges, plus one self‑loop per node
    src, dst = [], []
    for r in range(n_rows):
        for i in range(n_cols):
            for j in range(n_cols):
                if i != j:
                    src.append(r * n_cols + i); dst.append(r * n_cols + j)
    for c in range(n_cols):
        for i in range(n_rows):
            for j in range(n_rows):
                if i != j:
                    src.append(i * n_cols + c); dst.append(j * n_cols + c)
    num_cells = n_rows * n_cols
    for idx in range(num_cells):
        src.append(idx); dst.append(idx)

    edge_index = torch.stack([
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long)
    ], dim=0)

    return {'node_feats': fused, 'edge_index': edge_index}

def main():
    parser = argparse.ArgumentParser(description="Run StructRelNet MCM pretraining")
    parser.add_argument('--data_dir',    type=str, default='data/raw')
    parser.add_argument('--value_model', type=str, default='bert-base-uncased')
    parser.add_argument('--schema_model',type=str, default='bert-base-uncased')
    parser.add_argument('--epochs',      type=int, default=10)
    parser.add_argument('--lr',          type=float, default=5e-4)
    parser.add_argument('--mask_prob',   type=float, default=0.15)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--gnn_hidden',  type=int, default=128)
    parser.add_argument('--gnn_out',     type=int, default=128)
    parser.add_argument('--transformer_heads',  type=int, default=4)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dropout',     type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiations
    value_enc  = ValueEncoder(model_name=args.value_model, device=device)
    type_enc   = TypeEncoder(type_list=['cell','header','column','row'], embedding_dim=32)
    pos_enc    = PositionEncoder(dim=32)
    schema_enc = SchemaEncoder(model_name=args.schema_model, device=device)
    combiner   = CombineEmbeddings(fused_dim=args.gnn_hidden)

    # Build feature dicts for each table
    table_paths = sorted([
        os.path.join(args.data_dir, fn)
        for fn in os.listdir(args.data_dir)
        if os.path.isfile(os.path.join(args.data_dir, fn))
    ])
    print(f"Building graphs for {len(table_paths)} tables…")
    graph_dicts = [
        build_graph_data(p, value_enc, type_enc, pos_enc, schema_enc, combiner, device)
        for p in table_paths
    ]

    # Create dataset
    mcm_ds = MaskedCellModelingDataset(graph_dicts, mask_prob=args.mask_prob)

    # Initialize model
    model = StructRelNet(
        input_dim          = combiner.fused_dim,
        gnn_hidden         = args.gnn_hidden,
        gnn_out            = args.gnn_out,
        transformer_heads  = args.transformer_heads,
        transformer_layers = args.transformer_layers,
        gnn_type           = 'gcn',
        gnn_layers         = 2,
        dropout            = args.dropout,
        use_residual       = True
    ).to(device)

    optimizer = torch.optim.Adam(model.gnn.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop with robust sample parsing
    print("Starting MCM pretraining…")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for sample in mcm_ds:
            # Collect all tensors out of sample
            tensors = [x for x in sample if isinstance(x, torch.Tensor)]

            # Identify edge_index as the 2×E LongTensor
            edge_index = next(
                t for t in tensors
                if t.dtype == torch.long and t.ndim == 2
            ).to(device)

            # The two remaining float tensors are masked_feats & labels
            floats = [t for t in tensors if t.dtype.is_floating_point and t.ndim == 2]
            if len(floats) < 2:
                raise RuntimeError(f"Expected 2 float tensors, got {len(floats)}")
            masked_feats, labels = floats[0].to(device), floats[1].to(device)

            optimizer.zero_grad()
            out = model.gnn(masked_feats, edge_index)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(mcm_ds)
        print(f"Epoch {epoch}/{args.epochs} — Avg MCM loss: {avg_loss:.4f}")

    # Save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/pretrained_model.pt')
    print("✅ Saved pretrained model to checkpoints/pretrained_model.pt")

if __name__ == '__main__':
    main()


# python scripts/run_pretraining.py \
#   --data_dir data/raw \
#   --epochs 5 \
#   --lr 5e-4 \
#   --mask_prob 0.15