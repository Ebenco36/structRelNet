#!/usr/bin/env python
"""
scripts/run_train_cell_selector.py

Training entry point for the cell selection head:
1. Loads precomputed table features via TabFactDataset
2. Initializes StructRelNet with pretrained weights
3. Trains CellSelectionTrainer on train & val splits
4. Saves best cell selection checkpoint
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Core model and trainer
from model.structrelnet import StructRelNet
from fine_tuning.tabfact_dataset import TabFactDataset
from fine_tuning.trainer_cell_selection import CellSelectionTrainer


def main():
    parser = argparse.ArgumentParser(description="Run cell selection training")
    parser.add_argument('--train_meta',     type=str, default='data/processed/train_metadata.csv', help='Metadata CSV for train set')
    parser.add_argument('--train_feat_dir', type=str, default='data/processed/train_table_feats', help='Directory of train .pt feature files')
    parser.add_argument('--val_meta',       type=str, default='data/processed/val_metadata.csv', help='Metadata CSV for validation set')
    parser.add_argument('--val_feat_dir',   type=str, default='data/processed/val_table_feats', help='Directory of val .pt feature files')
    parser.add_argument('--pretrained_ckpt',type=str, default='checkpoints/pretrained_model.pt', help='Path to pretrained structrelnet')
    parser.add_argument('--output_ckpt',    type=str, default='checkpoints/best_cell_selector.pt', help='Where to save best cell selector')
    parser.add_argument('--batch_size',     type=int, default=8)
    parser.add_argument('--epochs',         type=int, default=5)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--seed',           type=int, default=42)
    args = parser.parse_args()

    # Reproducibility & device
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets
    train_ds = TabFactDataset(args.train_meta, args.train_feat_dir)
    val_ds   = TabFactDataset(args.val_meta,   args.val_feat_dir)

    # Model
    model = StructRelNet(input_dim=128).to(device)
    model.load_state_dict(torch.load(args.pretrained_ckpt, map_location=device))

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Trainer
    trainer = CellSelectionTrainer(model, optimizer, criterion, device)
    trainer.train(train_ds, val_ds, epochs=args.epochs, batch_size=args.batch_size)

    # Save final checkpoint
    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save(model.state_dict(), args.output_ckpt)
    print(f"Best cell-selection model saved to {args.output_ckpt}")

if __name__ == '__main__':
    main()
