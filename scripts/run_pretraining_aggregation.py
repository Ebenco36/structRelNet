#!/usr/bin/env python
"""
scripts/run_pretraining_aggregation.py

Aggregation pretraining for StructRelNet:
1. Load tables via TableLoaderFactory
2. (Optional) Build full StructRelGraph
3. Encode & fuse cell features (detached)
4. Build same‑row/column + self‑loops edge_index
5. Create one dummy or real agg_label per table
6. Train agg_head via custom single‑sample loop
7. Save pretrained aggregation weights
"""
import os
import sys
import argparse
import torch
import torch.nn as nn

# Ensure project root on PYTHONPATH
sys.path.append(os.getcwd())

from table_parser.table_factory      import TableLoaderFactory
from graph_builder.build_graph       import StructRelGraphBuilder
from embedding.value_encoder         import ValueEncoder
from embedding.type_encoder          import TypeEncoder
from embedding.position_encoder      import PositionEncoder
from embedding.schema_encoder        import SchemaEncoder
from embedding.combine_embeddings    import CombineEmbeddings
from model.structrelnet               import StructRelNet

def build_graph_agg(path, value_enc, type_enc, pos_enc, schema_enc, combiner, device):
    # 1) Load & graph‐build
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

    # 3) Encode
    schema_str  = ' '.join(table.header)
    schemas     = [schema_str] * len(values)
    val_emb     = value_enc(values).to(device)
    type_emb    = type_enc(types).to(device)
    pos_emb     = pos_enc(rows, cols).to(device)
    schema_emb  = schema_enc(schemas).to(device)

    # 4) Fuse & detach
    fused = combiner(val_emb, type_emb, pos_emb, schema_emb).detach().cpu()

    # 5) Build edges + self‑loops
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

    # 6) Dummy agg_label: single integer per table
    #    here we set all tables to class 0—replace with real labels if available
    agg_label = torch.tensor(0, dtype=torch.long)

    return fused, edge_index, agg_label

def main():
    parser = argparse.ArgumentParser("Run StructRelNet Aggregation pretraining")
    parser.add_argument('--data_dir',       type=str, default='data/raw')
    parser.add_argument('--value_model',    type=str, default='bert-base-uncased')
    parser.add_argument('--schema_model',   type=str, default='bert-base-uncased')
    parser.add_argument('--epochs',         type=int, default=10)
    parser.add_argument('--lr',             type=float, default=5e-4)
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--gnn_hidden',     type=int, default=128)
    parser.add_argument('--gnn_out',        type=int, default=128)
    parser.add_argument('--transformer_heads',  type=int, default=4)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--dropout',        type=float, default=0.1)
    parser.add_argument('--num_agg_classes', type=int, default=4)
    args = parser.parse_args()

    # reproducibility
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate encoders & combiner
    value_enc  = ValueEncoder(model_name=args.value_model,  device=device)
    type_enc   = TypeEncoder(type_list=['cell','header','column','row'],
                             embedding_dim=32)
    pos_enc    = PositionEncoder(dim=32)
    schema_enc = SchemaEncoder(model_name=args.schema_model, device=device)
    combiner   = CombineEmbeddings(fused_dim=args.gnn_hidden)

    # gather all tables
    files = sorted(os.listdir(args.data_dir))
    dataset = []
    for fn in files:
        path = os.path.join(args.data_dir, fn)
        if not os.path.isfile(path): continue
        dataset.append(
            build_graph_agg(path, value_enc, type_enc,
                            pos_enc, schema_enc, combiner, device)
        )

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
        num_agg_classes    = args.num_agg_classes
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # optimizer & loss
    optimizer = torch.optim.Adam(model.agg_head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("Starting Aggregation pretraining…")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for feats, eidx, agg_label in dataset:
            feats     = feats.to(device)
            eidx      = eidx.to(device)
            agg_label = agg_label.to(device)

            optimizer.zero_grad()
            # forward → _, _, agg_logits
            _, _, agg_logits = model(feats, eidx)
            loss = criterion(agg_logits.unsqueeze(0), agg_label.unsqueeze(0))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} — Avg Agg loss: {avg:.4f}")

    # save only the agg_head
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.agg_head.state_dict(), 'checkpoints/agg_head.pt')
    print("✅ Saved aggregation head to checkpoints/agg_head.pt")

if __name__ == '__main__':
    main()



# python scripts/run_pretraining_aggregation.py \
#   --data_dir data/raw \
#   --epochs 5 \
#   --lr 5e-4 \
#   --num_agg_classes 4
