# embedding/type_encoder.py
import torch
import torch.nn as nn

class TypeEncoder(nn.Module):
    def __init__(self, type_list=None, embedding_dim=64):
        super().__init__()
        self.type_to_idx = {t: i for i, t in enumerate(type_list or ['cell', 'header', 'column', 'row'])}
        self.embedding = nn.Embedding(len(self.type_to_idx), embedding_dim)

    def forward(self, types):
        indices = torch.tensor([self.type_to_idx.get(t, 0) for t in types], dtype=torch.long)
        return self.embedding(indices)
