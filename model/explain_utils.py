## model/explain_utils.py
import torch
from torch_geometric.nn import GNNExplainer
from captum.attr import IntegratedGradients

__all__ = ['explain_gnn', 'explain_transformer']


def explain_gnn(model, node_feats, edge_index, epochs: int = 20):
    """
    Wrapper to explain GNN predictions.
    Returns: feature_mask (N, F), edge_mask (E,)
    """
    explainer = GNNExplainer(model.gnn, epochs=epochs)
    return explainer.explain_graph(node_feats, edge_index)


def explain_transformer(model, node_feats, edge_index, target: str = 'aggregation'):
    """
    Integrated Gradients on transformer + head.
    Returns: attributions (N, D), convergence delta.
    """
    def forward_fn(x):
        g_out = model.gnn(x, edge_index)
        t_out = model.trans(g_out)
        logits = model.agg_head(t_out)
        return logits.unsqueeze(0)

    ig = IntegratedGradients(forward_fn)
    baseline = torch.zeros_like(node_feats)
    attributions, delta = ig.attribute(
        node_feats,
        baselines=baseline,
        return_convergence_delta=True
    )
    return attributions, delta