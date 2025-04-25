#!/usr/bin/env python
import sys
import os
sys.path.append(os.getcwd())
import json
import argparse
from table_parser.table_factory import TableLoaderFactory

def generate_metadata(raw_dir: str, out_path: str):
    """Generate train_metadata.json from raw table files, using full filename as table_id."""
    exts = ('.csv', '.tsv', '.xls', '.xlsx', '.html', '.htm')
    metadata = []

    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith(exts):
            continue
        table_id = fname               # <–– include extension for uniqueness
        path     = os.path.join(raw_dir, fname)

        # Load the table to infer shape
        table    = TableLoaderFactory.load(path)
        header   = table.header
        rows     = table.rows
        n_rows   = len(rows)
        n_cols   = len(header)

        # Default labels
        row_labels = [0] * n_rows
        col_labels = [0] * n_cols
        agg_label  = 0

        # Example real‐label rules
        if 'effect_size' in header:
            idx = header.index('effect_size')
            for i, row in enumerate(rows):
                try:
                    if float(row[idx]) > 1.0:
                        row_labels[i] = 1
                except:
                    pass
            col_labels[idx] = 1
            agg_label = 0
        elif 'age' in header and 'country' in header:
            idx = header.index('age')
            for i, row in enumerate(rows):
                try:
                    if int(row[idx]) > 30:
                        row_labels[i] = 1
                except:
                    pass
            col_labels[idx] = 1
            agg_label = 1
        # else: leave zeros

        metadata.append({
            'table_id':   table_id,
            'row_labels': row_labels,
            'col_labels': col_labels,
            'agg_label':  agg_label
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata written to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate metadata JSON (unique table_id includes extension)."
    )
    parser.add_argument(
        '--raw_dir', type=str, default='data/raw',
        help='Directory containing raw table files.'
    )
    parser.add_argument(
        '--out', type=str, default='data/processed/train_metadata.json',
        help='Output path for metadata JSON.'
    )
    args = parser.parse_args()
    generate_metadata(args.raw_dir, args.out)
