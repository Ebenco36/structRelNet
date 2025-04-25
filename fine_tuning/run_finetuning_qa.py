#!/usr/bin/env python
"""
Fine-tune StructRelNet on WikiTableQuestions for row selection
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# Add project root
sys.path.append(os.getcwd())

from scripts.qa_finetuning import QAFTTrainer, WikiTableQADataset
from table_parser.table_factory import TableLoaderFactory
from graph_builder.build_graph import StructRelGraphBuilder
from embedding.value_encoder import ValueEncoder
from embedding.type_encoder import TypeEncoder
from embedding.position_encoder import PositionEncoder
from embedding.schema_encoder import SchemaEncoder
from embedding.combine_embeddings import CombineEmbeddings
from model.structrelnet import StructRelNet

def build_graph_sample(path, value_enc, type_enc, pos_enc, schema_enc, combiner, device):
    table = TableLoaderFactory.load(path)
    StructRelGraphBuilder().build(table)

    n_rows, n_cols = len(table.rows), len(table.header)
    values, types, rows, cols = [], [], [], []
    for r, row in enumerate(table.rows):
        for c, v in enumerate(row):
            values.append(v)
            types.append(table.types[c] if table.types else "cell")
            rows.append(r)
            cols.append(c)

    schemas = [' '.join(table.header)] * len(values)
    val_emb = value_enc(values).to(device)
    typ_emb = type_enc(types).to(device)
    pos_emb = pos_enc(rows, cols).to(device)
    sch_emb = schema_enc(schemas).to(device)

    fused = combiner(val_emb, typ_emb, pos_emb, sch_emb).detach().cpu()

    src, dst = [], []
    for r in range(n_rows):
        for i in range(n_cols):
            for j in range(n_cols):
                if i != j:
                    src.append(r * n_cols + i)
                    dst.append(r * n_cols + j)
    for c in range(n_cols):
        for i in range(n_rows):
            for j in range(n_rows):
                if i != j:
                    src.append(i * n_cols + c)
                    dst.append(j * n_cols + c)
    for i in range(n_rows * n_cols):
        src.append(i)
        dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Dummy: mark row 0 as correct
    target = torch.zeros(n_rows, dtype=torch.float32)
    target[0] = 1.0

    return {"node_feats": fused, "edge_index": edge_index, "target_rows": target.unsqueeze(0)}

def main():
    p = argparse.ArgumentParser("Fine-tune StructRelNet on WikiTableQuestions")
    p.add_argument('--data_dir', type=str, default='data/raw')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--gnn_hidden', type=int, default=128)
    p.add_argument('--gnn_out', type=int, default=128)
    p.add_argument('--transformer_heads', type=int, default=4)
    p.add_argument('--transformer_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Encoders
    value_enc = ValueEncoder(device=device)
    type_enc = TypeEncoder(type_list=["cell", "header", "column", "row"], embedding_dim=32)
    pos_enc = PositionEncoder(dim=32)
    schema_enc = SchemaEncoder(device=device)
    combiner = CombineEmbeddings(fused_dim=args.gnn_hidden)

    # Build graph inputs
    paths = sorted([os.path.join(args.data_dir, fn) for fn in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, fn))])
    graphs = [build_graph_sample(p, value_enc, type_enc, pos_enc, schema_enc, combiner, device) for p in paths]

    # Model
    model = StructRelNet(
        input_dim=combiner.fused_dim,
        gnn_hidden=args.gnn_hidden,
        gnn_out=args.gnn_out,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.row_head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    trainer = QAFTTrainer(model.gnn, model.row_head, optimizer, criterion, device)
    trainer.train(WikiTableQADataset(graphs), epochs=args.epochs, batch_size=4)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.row_head.state_dict(), 'checkpoints/row_head_finetuned.pt')
    print("✅ Saved fine-tuned row head to checkpoints/row_head_finetuned.pt")

if __name__ == '__main__':
    main()
