import torch
import torch.nn as nn
from model.gnn_module import GNNModule
from model.transformer_utils import TableTransformer

class StructRelNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        gnn_hidden: int = 128,
        gnn_out: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        gnn_type: str = 'gcn',
        gnn_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,       # Controls residuals in the GNN
        num_agg_classes: int = 4         # Number of aggregation classes
    ):
        super().__init__()

        # 🔹 GNN backbone
        self.gnn = GNNModule(
            input_dim=input_dim,
            hidden_dim=gnn_hidden,
            output_dim=gnn_out,
            gnn_type=gnn_type,
            num_layers=gnn_layers,
            dropout=dropout,
            use_residual=use_residual
        )

        # 🔹 TableTransformer stack on the raw input features
        self.transformers = nn.ModuleList([
            TableTransformer(
                embed_dim=input_dim,
                num_heads=transformer_heads,
                dropout=dropout
            )
            for _ in range(transformer_layers)
        ])

        # Column‑selection head: one logit per column
        self.col_head = nn.Linear(gnn_out, 1)

        # 🔹 Row‑selection head: binary logit per node (row)
        self.row_head = nn.Linear(gnn_out, 1)

        # 🔹 Aggregation head: multi‑class logit per table
        self.agg_head = nn.Linear(gnn_out, num_agg_classes)

        # 🔹 Fusion & general output head (optional)
        fused_size = input_dim + gnn_out
        self.fusion_layer = nn.Linear(fused_size, 256)
        self.output_layer = nn.Linear(256, 1)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor):
        """
        Args:
            node_feats: [N, input_dim] — your fused per-cell embeddings
            edge_index: [2, E] — cell–cell edges
        Returns:
            main_logits: [N]       — optional unified output per node
            row_logits: [N]        — binary selection logit per node
            agg_logits: [num_cls]  — multi‑class aggregation logit per table
        """
        # 1) GNN over the cell graph
        gnn_out = self.gnn(node_feats, edge_index)           # [N, gnn_out]

        # 2) TableTransformer on the raw node_feats
        x = node_feats.unsqueeze(0)                          # [1, N, input_dim]
        for tbl_tr in self.transformers:
            x = tbl_tr(x)                                    # still [1, N, input_dim]
        trans_out = x.squeeze(0)                             # [N, input_dim]

        # 3) General fusion output (if you still need it)
        fused = torch.cat([gnn_out, trans_out], dim=-1)      # [N, fused_size]
        fused = torch.relu(self.fusion_layer(fused))         # [N, 256]
        main_logits = self.output_layer(fused).squeeze(-1)   # [N]

        # 4) Row‑selection: one logit per node
        row_logits = self.row_head(gnn_out).squeeze(-1)      # [N]

        # 5) Aggregation: pool over nodes then classify
        # (you can swap to mean/max as desired)
        graph_repr = gnn_out.mean(dim=0)                     # [gnn_out]
        agg_logits = self.agg_head(graph_repr)               # [num_agg_classes]

        return main_logits, row_logits, agg_logits
