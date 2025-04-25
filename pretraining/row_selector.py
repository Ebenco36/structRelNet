## pretraining/row_selector.py
import torch
from torch.utils.data import Dataset

class RowSelectionDataset(Dataset):
    """
    Dataset for Row Selection head.
    Labels: binary vector per node indicating target row.
    """
    def __init__(self, table_graphs: list):
        self.graphs = table_graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        return data['node_feats'], data['edge_index'], data['row_labels']


class RowSelectorTrainer:
    """
    Trainer for row selection objective.
    """
    def __init__(self, gnn_module, row_head, optimizer, criterion, device):
        self.gnn = gnn_module.to(device)
        self.row_head = row_head.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, dataset, epochs=5, batch_size=16):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.gnn.train(); self.row_head.train()
        for epoch in range(1, epochs+1):
            total_loss = 0.0
            for feats, edge_idx, labels in loader:
                feats = feats.to(self.device)
                edge_idx = edge_idx.to(self.device)
                labels = labels.to(self.device)
                out = self.gnn(feats, edge_idx)
                logits = self.row_head(out)
                loss = self.criterion(logits, labels.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}/{epochs} Row Loss: {total_loss/len(loader):.4f}")