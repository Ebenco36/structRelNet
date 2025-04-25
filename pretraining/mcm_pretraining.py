## pretraining/mcm_pretraining.py
import torch
from torch.utils.data import Dataset

class MaskedCellModelingDataset(Dataset):
    """
    Dataset for Masked Cell Modeling (MCM).
    Masks a fraction of node features and predicts the original values.
    """
    def __init__(self, table_graphs: list, mask_prob: float = 0.15):
        self.graphs = table_graphs  # list of dicts with 'node_feats', 'edge_index'
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        node_feats = data['node_feats']  # Tensor [N, D]
        edge_index = data['edge_index']  # LongTensor [2, E]
        # Create mask
        mask = (torch.rand(node_feats.size(0)) < self.mask_prob)
        masked_feats = node_feats.clone()
        masked_feats[mask] = 0
        return masked_feats, node_feats, edge_index, mask


class MCMTrainer:
    """
    Trainer for masked cell modeling objective.
    """
    def __init__(self, gnn_module, optimizer, criterion, device):
        self.gnn = gnn_module.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, dataset, epochs=5, batch_size=16):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.gnn.train()
        for epoch in range(1, epochs+1):
            total_loss = 0.0
            for masked_feats, orig_feats, edge_idx, mask in loader:
                masked_feats = masked_feats.to(self.device)
                orig_feats = orig_feats.to(self.device)
                edge_idx = edge_idx.to(self.device)
                mask = mask.to(self.device)
                out = self.gnn(masked_feats, edge_idx)
                # MSE on masked positions
                loss = self.criterion(out[mask], orig_feats[mask])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}/{epochs} MCM Loss: {total_loss/len(loader):.4f}")