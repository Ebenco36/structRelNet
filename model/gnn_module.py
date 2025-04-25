import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.norm import GraphNorm

class GNNModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gnn_type: str = 'gcn',
        num_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
        use_residual: bool = None,
    ):
        super().__init__()
        # allow alias
        self.residual = use_residual if use_residual is not None else residual

        self.layers = nn.ModuleList()
        self.norms  = nn.ModuleList()
        self.dropout    = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.gnn_type   = gnn_type.lower()

        if self.gnn_type == 'gcn':
            # no self‐loops, no gcn_norm
            self.layers.append(GCNConv(input_dim, hidden_dim,
                                       add_self_loops=False,
                                       normalize=False))
            self.norms.append(GraphNorm(hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim,
                                           add_self_loops=False,
                                           normalize=False))
                self.norms.append(GraphNorm(hidden_dim))
            self.layers.append(GCNConv(hidden_dim, output_dim,
                                       add_self_loops=False,
                                       normalize=False))
            self.norms.append(GraphNorm(output_dim))

        elif self.gnn_type == 'gat':
            self.layers.append(GATConv(input_dim, hidden_dim, heads=2, concat=True))
            self.norms.append(GraphNorm(hidden_dim * 2))
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_dim * 2, hidden_dim, heads=1))
                self.norms.append(GraphNorm(hidden_dim))
            self.layers.append(GATConv(hidden_dim, output_dim, heads=1))
            self.norms.append(GraphNorm(output_dim))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.layers, self.norms):
            identity = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            if self.residual and identity.shape == x.shape:
                x = x + identity
        return x
