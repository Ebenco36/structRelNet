#!/usr/bin/env python
"""
scripts/prepare_data.py

Precompute table features for fine‑tuning:
- Loads pretrained StructRelNet encoder
- Uses TableFactory to load tables
- Builds StructRel graph per table
- Encodes each table through GNN + Transformer
- Saves dictionaries of node_feats and edge_index to .pt files
"""
import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch

from table_parser.table_loader import load_table
from graph_builder.build_graph import StructRelGraphBuilder

from embedding.value_encoder import ValueEncoder
from embedding.type_encoder import TypeEncoder
from embedding.position_encoder import PositionEncoder
from embedding.schema_encoder import SchemaEncoder
from embedding.combine_embeddings import CombineEmbeddings

from model.structrelnet import StructRelNet


def process_split(
    split_dir: str,
    out_dir: str,
    model: StructRelNet,
    value_enc: ValueEncoder,
    type_enc: TypeEncoder,
    pos_enc: PositionEncoder,
    schema_enc: SchemaEncoder,
    combiner: CombineEmbeddings,
    builder: StructRelGraphBuilder,
    device: torch.device
):
    os.makedirs(out_dir, exist_ok=True)
    table_files = [f for f in os.listdir(split_dir)
                   if os.path.isfile(os.path.join(split_dir, f))]
    print(f"Processing {len(table_files)} tables in '{split_dir}' → '{out_dir}'")
    model.eval()
    with torch.no_grad():
        for fname in table_files:
            path = os.path.join(split_dir, fname)
            try:
                table = load_table(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue

            graph = builder.build(table)

            # Gather per-cell attributes
            values = [cell.value for cell in table.cells]
            types_ = [cell.cell_type for cell in table.cells]
            rows   = [cell.row for cell in table.cells]
            cols   = [cell.col for cell in table.cells]
            schemas = [getattr(table, 'schema_str', '')] * len(table.cells)

            # Encode modalities
            val_emb    = value_enc(values).to(device)
            type_emb   = type_enc(types_).to(device)
            pos_emb    = pos_enc(rows, cols).to(device)
            schema_emb = schema_enc(schemas).to(device)

            # Fuse features
            fused = combiner(val_emb, type_emb, pos_emb, schema_emb)
            # GNN + Transformer encoding
            edge_index = graph['edge_index'].to(device)
            gnn_out    = model.gnn(fused, edge_index)
            trans_out  = model.trans(gnn_out)

            # Final node features
            node_feats = trans_out.cpu()
            edge_index = edge_index.cpu()

            # Prepare save dict
            save_dict = {'node_feats': node_feats, 'edge_index': edge_index}
            if 'row_labels' in graph:
                save_dict['row_labels'] = graph['row_labels']
            if 'agg_label' in graph:
                save_dict['agg_label'] = graph['agg_label']

            # Save to .pt
            table_id = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{table_id}.pt")
            torch.save(save_dict, out_path)
    print(f"Finished split: '{split_dir}'\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare table features for fine‑tuning"
    )
    parser.add_argument(
        '--pretrained_ckpt',
        type=str,
        default='checkpoints/pretrained_model.pt',
        help='Path to pretrained StructRelNet checkpoint'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data/processed',
        help='Root directory containing processed table splits'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train_tables', 'val_tables', 'test_tables'],
        help='Names of subfolders under data_root to process'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='data/processed',
        help='Root directory for output feature folders'
    )
    parser.add_argument(
        '--value_model',
        type=str,
        default='bert-base-uncased',
        help='Transformers model name for value encoder'
    )
    parser.add_argument(
        '--schema_model',
        type=str,
        default='bert-base-uncased',
        help='Transformers model name for schema encoder'
    )
    parser.add_argument(
        '--type_dim',
        type=int,
        default=32,
        help='Embedding dimension for type encoder'
    )
    parser.add_argument(
        '--pos_dim',
        type=int,
        default=32,
        help='Embedding dimension for position encoder'
    )
    parser.add_argument(
        '--fused_dim',
        type=int,
        default=128,
        help='Dimension of fused embeddings'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for computation (cuda or cpu)'
    )
    args = parser.parse_args()

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Check pretrained checkpoint
    if not os.path.isfile(args.pretrained_ckpt):
        print(f"Error: pretrained checkpoint '{args.pretrained_ckpt}' not found. Run pretraining first.")
        return
    # Load StructRelNet
    model = StructRelNet(input_dim=args.fused_dim).to(device)
    checkpoint = torch.load(args.pretrained_ckpt, map_location=device)
    model.load_state_dict(checkpoint)

    # Instantiate encoders and graph builder
    value_enc  = ValueEncoder(model_name=args.value_model, device=device)
    type_enc   = TypeEncoder(type_list=['cell','header','column','row'], embedding_dim=args.type_dim)
    pos_enc    = PositionEncoder(dim=args.pos_dim)
    schema_enc = SchemaEncoder(model_name=args.schema_model, device=device)
    combiner   = CombineEmbeddings(fused_dim=args.fused_dim).to(device)
    builder    = StructRelGraphBuilder()

    # Ensure output root exists
    os.makedirs(args.output_root, exist_ok=True)
    raw_root = 'data/raw'

    for split in args.splits:
        proc_dir = os.path.join(args.data_root, split)
        if os.path.isdir(proc_dir):
            split_dir = proc_dir
        elif os.path.isdir(raw_root):
            print(f"Processed split '{proc_dir}' not found, using raw folder '{raw_root}'")
            split_dir = raw_root
        else:
            print(f"Warning: neither '{proc_dir}' nor '{raw_root}' exist. Skipping split '{split}'.")
            continue

        out_dir = os.path.join(
            args.output_root,
            split.replace('_tables', '_table_feats')
        )
        process_split(
            split_dir, out_dir,
            model, value_enc, type_enc, pos_enc,
            schema_enc, combiner, builder, device
        )

if __name__ == '__main__':
    main()


# python scripts/prepare_data.py \
#   --pretrained_ckpt checkpoints/pretrained_model.pt \
#   --data_root data/processed \
#   --splits train_tables val_tables test_tables \
#   --output_root data/processed \
#   --value_model bert-base-uncased \
#   --schema_model bert-base-uncased \
#   --type_dim 32 \
#   --pos_dim 32 \
#   --fused_dim 128 \
#   --device cuda
