#!/usr/bin/env python
"""
scripts/explainability.py

Use GNNExplainer to find the most important nodes/edges for a specific
row‑selection or aggregation decision on a single table, on CPU.
"""
import os
import sys
import json
import argparse
import torch

# Try all known locations for GNNExplainer in PyG
try:
    from torch_geometric.nn import GNNExplainer
except ImportError:
    try:
        from torch_geometric.nn.explain import GNNExplainer
    except ImportError:
        from torch_geometric.explain import GNNExplainer

from torch_geometric.data import Data

sys.path.append(os.getcwd())

from table_parser.table_factory    import TableLoaderFactory
from embedding.value_encoder       import ValueEncoder
from embedding.type_encoder        import TypeEncoder
from embedding.position_encoder    import PositionEncoder
from embedding.schema_encoder      import SchemaEncoder
from embedding.combine_embeddings  import CombineEmbeddings
from model.structrelnet            import StructRelNet

def build_pyg_data(path, value_enc, type_enc, pos_enc, schema_enc, combiner, device):
    table = TableLoaderFactory.load(path)
    n_rows, n_cols = len(table.rows), len(table.header)

    vals, types, rows, cols = [], [], [], []
    for r in range(n_rows):
        for c in range(n_cols):
            vals.append(table.rows[r][c])
            types.append(table.types[c] if table.types else 'unknown')
            rows.append(r); cols.append(c)

    schema_str = ' '.join(table.header)
    schemas    = [schema_str] * len(vals)

    v_emb = value_enc(vals).to(device)
    t_emb = type_enc(types).to(device)
    p_emb = pos_enc(rows, cols).to(device)
    s_emb = schema_enc(schemas).to(device)
    x     = combiner(v_emb, t_emb, p_emb, s_emb)

    # build edges
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
    for idx in range(n_rows * n_cols):
        src.append(idx); dst.append(idx)

    edge_index = torch.stack([
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long)
    ], dim=0).to(device)

    return Data(x=x, edge_index=edge_index), n_rows, n_cols, table.header

def parse_args():
    p = argparse.ArgumentParser("GNNExplainer for StructRelNet")
    p.add_argument('--table',      type=str, required=True,
                   help="Path to a single table (csv/tsv/…) to explain")
    p.add_argument('--head',       choices=['row','col','agg'], default='row',
                   help="Which head to explain")
    p.add_argument('--metadata',   type=str, default='data/processed/train_metadata.json',
                   help="metadata JSON for label lookup")
    p.add_argument('--topk_nodes', type=int, default=10,
                   help="Number of top nodes to print")
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device('cpu')

    # Load metadata
    with open(args.metadata) as f:
        meta_list = json.load(f)
    meta = { e['table_id']: e for e in meta_list }
    table_id = os.path.basename(args.table)
    if table_id not in meta:
        print(f"No metadata for '{table_id}'"); sys.exit(1)
    entry = meta[table_id]

    # Encoders + combiner
    value_enc  = ValueEncoder(device=device)
    type_enc   = TypeEncoder(type_list=['cell','header','column','row'], embedding_dim=32)
    pos_enc    = PositionEncoder(dim=32)
    schema_enc = SchemaEncoder(device=device)
    combiner   = CombineEmbeddings(fused_dim=128)

    # Load pretrained model
    state = torch.load('checkpoints/pretrained_model.pt', map_location='cpu')
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

    # Load fine‑tuned head
    head_ckpt = f"checkpoints/{args.head}_finetuned.pt"
    if os.path.isfile(head_ckpt):
        getattr(model, f"{args.head}_head").load_state_dict(
            torch.load(head_ckpt, map_location='cpu'),
            strict=False
        )
    model.eval()

    # Prepare data
    data, n_rows, n_cols, header = build_pyg_data(
        args.table, value_enc, type_enc, pos_enc, schema_enc, combiner, device
    )

    # Decide target index
    if args.head == 'agg':
        target = None
    else:
        labels = entry[f'{args.head}_labels']
        target = labels.index(1) if 1 in labels else 0

    # Instantiate explainer
    explainer = GNNExplainer(model.gnn, 200)

    # Explain
    if hasattr(explainer, 'explain_graph'):
        if args.head == 'agg':
            def fn(x, edge_index): return model(x, edge_index)[2]
            node_mask, edge_mask = explainer.explain_graph(fn, data.x, data.edge_index)
        else:
            def fn(x, edge_index):
                feat = model.gnn(x, edge_index)
                emb  = feat.view(n_rows, n_cols, -1).mean(dim=1 if args.head=='row' else 0)
                return getattr(model, f"{args.head}_head")(emb).squeeze(-1)[target].unsqueeze(0)
            node_mask, edge_mask = explainer.explain_graph(fn, data.x, data.edge_index, target=0)
    else:
        # Fallback to .explain()
        if args.head == 'agg':
            node_mask, edge_mask = explainer.explain(data.x, data.edge_index)
        else:
            node_mask, edge_mask = explainer.explain(data.x, data.edge_index)

    # Print top‑k nodes
    scores = node_mask.abs()
    topk   = torch.topk(scores, args.topk_nodes)[1].tolist()
    print(f"\nTop {args.topk_nodes} influential nodes for '{args.head}':")
    for idx in topk:
        r, c = divmod(idx, n_cols)
        print(f"  Node {idx} → row {r}, col {c}, header='{header[c]}'")

if __name__ == "__main__":
    main()
