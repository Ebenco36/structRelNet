import os
import torch
import networkx as nx
from pyvis.network import Network

def visualize_predictions(nx_graph, predictions, output_file="outputs/predicted_graph.html"):
    """
    Visualizes StructRelGraph with predicted node labels.
    
    Args:
        nx_graph: NetworkX graph
        predictions: list or tensor of predicted probabilities (same order as nx_graph.nodes)
        output_file: path to save HTML
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    for i, (node, data) in enumerate(nx_graph.nodes(data=True)):
        label = data.get("value") or data.get("name") or str(node)
        node_type = data.get("node_type", "unknown")

        pred = predictions[i].item() if isinstance(predictions[i], torch.Tensor) else predictions[i]
        color = "lightgreen" if pred >= 0.5 else "salmon"

        net.add_node(
            node,
            label=label,
            title=f"{node_type} | Pred: {pred:.2f}",
            color=color,
        )

    for u, v, d in nx_graph.edges(data=True):
        edge_type = d.get("type", "relation")
        net.add_edge(u, v, label=edge_type, title=edge_type)

    # Use write_html to avoid notebook-only rendering errors
    net.write_html(output_file, notebook=False)
    print(f"[✔] Prediction graph saved: {output_file}")
