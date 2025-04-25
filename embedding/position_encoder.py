# embedding/position_encoder.py
import torch
import torch.nn as nn

class PositionEncoder(nn.Module):
    def __init__(self, max_row=1000, max_col=100, dim=64):
        super().__init__()
        self.row_embedding = nn.Embedding(max_row, dim)
        self.col_embedding = nn.Embedding(max_col, dim)

    def forward(self, row_indices, col_indices):
        row_indices = torch.tensor(row_indices, dtype=torch.long)
        col_indices = torch.tensor(col_indices, dtype=torch.long)
        return self.row_embedding(row_indices) + self.col_embedding(col_indices)

