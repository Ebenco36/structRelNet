## pretraining/aggregation_head.py
import torch
from torch.utils.data import Dataset

class AggregationDataset(Dataset):
    """
    Dataset for aggregation-type classification.
    Labels: integer class per table.
    """
    def __init__(self, table_graphs: list):
        self.graphs = table_graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        return data['node_feats'], data['edge_index'], data['agg_label']


class AggregationTrainer:
    """
    Trainer for aggregation classification.
    """
    def __init__(self, gnn_module, transformer, agg_head, optimizer, criterion, device):
        self.gnn = gnn_module.to(device)
        self.transformer = transformer.to(device)
        self.agg_head = agg_head.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, dataset, epochs=5, batch_size=16):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.gnn.train(); self.transformer.train(); self.agg_head.train()
        for epoch in range(1, epochs+1):
            total_loss = 0.0
            for feats, edge_idx, labels in loader:
                feats = feats.to(self.device)
                edge_idx = edge_idx.to(self.device)
                labels = labels.to(self.device)
                g_out = self.gnn(feats, edge_idx)
                t_out = self.transformer(g_out)
                logits = self.agg_head(t_out)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}/{epochs} Agg Loss: {total_loss/len(loader):.4f}")