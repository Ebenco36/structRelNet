import torch
from torch.utils.data import Dataset, DataLoader

class WikiTableQADataset(Dataset):
    """
    Dataset for fine-tuning StructRelNet on table QA tasks (e.g., WikiTableQuestions).
    Each sample contains fused node features, edge index, and row-level binary labels.
    """
    def __init__(self, graph_data_list):
        self.graphs = graph_data_list

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        sample = self.graphs[idx]
        return sample['node_feats'], sample['edge_index'], sample['target_rows']

class QAFTTrainer:
    """
    Fine-tuning trainer for StructRelNet on row selection (e.g., answer row prediction).
    """
    def __init__(self, gnn, row_head, optimizer, criterion, device):
        self.gnn = gnn.to(device)
        self.row_head = row_head.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, dataset, epochs=5, batch_size=1):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
        self.gnn.train()
        self.row_head.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in loader:
                for feats, edge_index, targets in batch:
                    feats = feats.to(self.device)
                    edge_index = edge_index.to(self.device)
                    targets = targets.to(self.device)

                    node_out = self.gnn(feats, edge_index)

                    N = node_out.size(0)
                    R = targets.size(1)
                    C = N // R
                    row_emb = node_out.view(R, C, -1).mean(dim=1)

                    logits = self.row_head(row_emb).squeeze(-1)
                    loss = self.criterion(logits, targets.squeeze(0).float())


                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

            print(f"Epoch {epoch}/{epochs} — QA fine-tune loss: {total_loss / len(dataset):.4f}")

