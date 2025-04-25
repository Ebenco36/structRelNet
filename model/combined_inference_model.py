import torch
import torch.nn as nn
from model.structrelnet import StructRelNet

class CombinedQAInferenceModel(nn.Module):
    def __init__(
        self,
        input_dim,
        gnn_hidden,
        gnn_out,
        transformer_heads,
        transformer_layers,
        gnn_type,
        gnn_layers,
        dropout,
        num_agg_classes,
        pretrained_paths,
        device="cpu"
    ):
        super().__init__()

        # Initialize full StructRelNet architecture
        self.model = StructRelNet(
            input_dim=input_dim,
            gnn_hidden=gnn_hidden,
            gnn_out=gnn_out,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
            gnn_type=gnn_type,
            gnn_layers=gnn_layers,
            dropout=dropout,
            use_residual=True,
            num_agg_classes=num_agg_classes
        ).to(device)

        # ✅ Load GNN backbone
        if "mcm" in pretrained_paths:
            print("🔁 Loading GNN weights from MCM pretraining...")
            gnn_state = torch.load(pretrained_paths["mcm"], map_location=device)
            self.model.gnn.load_state_dict(gnn_state)

        # ✅ Load row selector head
        if "row" in pretrained_paths:
            print("🔁 Loading Row Selector Head...")
            row_state = torch.load(pretrained_paths["row"], map_location=device)
            self.model.row_head.load_state_dict(row_state)

        # ✅ Load aggregation head
        if "agg" in pretrained_paths:
            print("🔁 Loading Aggregation Head...")
            agg_state = torch.load(pretrained_paths["agg"], map_location=device)
            self.model.agg_head.load_state_dict(agg_state)

    def forward(self, node_feats, edge_index):
        return self.model(node_feats, edge_index)
