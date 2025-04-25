## pretraining/multitask_trainer.py
import torch
from torch.utils.data import DataLoader

class MultiTaskTrainer:
    """
    Jointly trains MCM, Row Selection, and Aggregation objectives.
    """
    def __init__(
        self,
        gnn_module,
        transformer,
        row_head,
        agg_head,
        optimizers: dict,
        criterions: dict,
        device
    ):
        self.gnn = gnn_module.to(device)
        self.transformer = transformer.to(device)
        self.row_head = row_head.to(device)
        self.agg_head = agg_head.to(device)
        self.opts = optimizers
        self.crits = criterions
        self.device = device

    def train(self, datasets: dict, epochs=5, batch_size=8):
        loaders = {name: DataLoader(ds, batch_size=batch_size, shuffle=True)
                   for name, ds in datasets.items()}
        for epoch in range(1, epochs+1):
            losses = {name: 0.0 for name in loaders}
            for batch_mcm, batch_row, batch_agg in zip(*loaders.values()):
                # Zero grads
                for opt in self.opts.values(): opt.zero_grad()
                # MCM
                m_in, m_orig, m_eidx, m_mask = [x.to(self.device) for x in batch_mcm]
                m_out = self.gnn(m_in, m_eidx)
                loss_m = self.crits['mcm'](m_out[m_mask], m_orig[m_mask])
                # Row
                r_feats, r_eidx, r_lbl = [x.to(self.device) for x in batch_row]
                r_out = self.gnn(r_feats, r_eidx)
                r_logits = self.row_head(r_out)
                loss_r = self.crits['row'](r_logits, r_lbl.float())
                # Agg
                a_feats, a_eidx, a_lbl = [x.to(self.device) for x in batch_agg]
                a_out = self.gnn(a_feats, a_eidx)
                a_trans = self.transformer(a_out)
                a_logits = self.agg_head(a_trans)
                loss_a = self.crits['agg'](a_logits, a_lbl)
                # Total
                loss = loss_m + loss_r + loss_a
                loss.backward()
                for opt in self.opts.values(): opt.step()
                # Accumulate
                losses['mcm'] += loss_m.item()
                losses['row'] += loss_r.item()
                losses['agg'] += loss_a.item()
            print(
                f"Epoch {epoch}/{epochs} "
                f"MCM: {losses['mcm']/len(loaders['mcm']):.4f}, "
                f"Row: {losses['row']/len(loaders['row']):.4f}, "
                f"Agg: {losses['agg']/len(loaders['agg']):.4f}"
            )