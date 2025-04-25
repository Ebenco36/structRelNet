# embedding/combine_embeddings.py
import torch
import torch.nn as nn

class CombineEmbeddings(nn.Module):
    def __init__(self, fused_dim=256):
        super().__init__()
        self.fused_dim = fused_dim
        self.proj = None  # Init later dynamically
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, *embeddings):
        x = torch.cat(embeddings, dim=-1)
        if self.proj is None:
            # Initialize projection layer based on actual input size
            self.proj = nn.Linear(x.size(-1), self.fused_dim).to(x.device)
        return self.activation(self.proj(self.dropout(x)))
