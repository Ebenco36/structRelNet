#!/usr/bin/env python
"""
scripts/run_inference_qa.py

Run question answering inference on a table dataset using a fine-tuned StructRelNet.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import ast
from pathlib import Path

# Project root on PYTHONPATH
sys.path.append(os.getcwd())

from table_parser.table_factory import TableLoaderFactory
from schema_inference.infer_schema import SchemaInferer
from graph_builder.build_graph import StructRelGraphBuilder
from embedding.value_encoder import ValueEncoder
from embedding.type_encoder import TypeEncoder
from embedding.position_encoder import PositionEncoder
from embedding.schema_encoder import SchemaEncoder
from embedding.combine_embeddings import CombineEmbeddings
from model.structrelnet import StructRelNet

def extract_answer_cells(table):
    """
    Flatten table to a list of (row_index, row_values)
    """
    return [row for row in table.rows]

def find_answer_match(pred_row, gold_answers):
    for cell in pred_row:
        for gold in gold_answers:
            if gold.lower().strip() in str(cell).lower():
                return True
    return False

def infer_single(table_path, question, model, encoders, device):
    table = TableLoaderFactory.load(table_path)
    table = SchemaInferer().infer(table)
    StructRelGraphBuilder().build(table)

    # Flatten cells
    n_rows, n_cols = len(table.rows), len(table.header)
    values, types, rows, cols = [], [], [], []
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row):
            values.append(cell)
            types.append(table.types[c_idx] if table.types else 'unknown')
            rows.append(r_idx)
            cols.append(c_idx)

    schema_str = " ".join(table.header)
    schemas = [schema_str] * len(values)

    # Encode features
    val_emb = encoders["value"](values).to(device)
    type_emb = encoders["type"](types).to(device)
    pos_emb = encoders["position"](rows, cols).to(device)
    schema_emb = encoders["schema"](schemas).to(device)

    fused = encoders["combine"](val_emb, type_emb, pos_emb, schema_emb)
    fused = fused.detach()

    # Build edge_index
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
    for idx in range(n_rows * n_cols):
        src.append(idx)
        dst.append(idx)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Inference
    model.eval()
    with torch.no_grad():
        node_logits, _, _ = model(fused.to(device), edge_index.to(device))

    # Aggregate to rows
    node_logits = node_logits.view(n_rows, n_cols).mean(dim=1)  # [n_rows]
    pred_row_idx = torch.argmax(node_logits).item()
    pred_row = table.rows[pred_row_idx]

    return pred_row_idx, pred_row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing tables")
    parser.add_argument("--metadata_path", type=str, required=True, help="CSV with id, path, meta[question, answers]")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoders
    encoders = {
        "value": ValueEncoder(device=device),
        "type": TypeEncoder(type_list=["cell", "header", "column", "row"], embedding_dim=32),
        "position": PositionEncoder(dim=32),
        "schema": SchemaEncoder(device=device),
        "combine": CombineEmbeddings(fused_dim=128)
    }

    # Load model
    print("🔁 Loading GNN weights from MCM pretraining...")
    model = StructRelNet(
        input_dim=128,
        gnn_hidden=128,
        gnn_out=128,
        transformer_heads=4,
        transformer_layers=2,
        gnn_type="gcn",
        gnn_layers=2,
        dropout=0.1,
        use_residual=True,
        num_agg_classes=4
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/pretrained_model.pt", map_location=device))

    # Load fine-tuned row head
    print("🔁 Loading row selector head...")
    model.row_head.load_state_dict(torch.load("checkpoints/row_selector.pt", map_location=device))

    # Read metadata
    df = pd.read_csv(args.metadata_path)
    correct = 0

    print("\n🧠 Running QA inference...\n")
    for i, row in df.iterrows():
        meta = ast.literal_eval(row["meta"])
        question = meta["question"]
        gold_answers = meta["answers"]

        table_path = Path(args.data_dir) / Path(row["path"]).name
        pred_idx, pred_row = infer_single(table_path, question, model, encoders, device)

        print(f"🔎 Question: {question}")
        print(f"✅ Answer:   {gold_answers[0]}")
        print(f"📌 Predicted Row [{pred_idx}]: {pred_row}")

        if find_answer_match(pred_row, gold_answers):
            print("✅ Match: ✅\n")
            correct += 1
        else:
            print("✅ Match: ❌\n")

    total = len(df)
    print(f"✅ QA Inference Accuracy: {correct}/{total} = {correct / total:.2%}")

if __name__ == "__main__":
    main()
