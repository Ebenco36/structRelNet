# graph_builder/visualize_graph_pyvis.py

import os
import json
import torch
import webbrowser
import networkx as nx
from pyvis.network import Network
from torch_geometric.utils import from_networkx


def visualize_graph_pyvis(G: nx.Graph, output_file: str):
    """
    Export a StructRelGraph to different formats based on file extension.
    Supports: .html (interactive), .json, .graphml, .edgelist, .pt (PyG)
    """

    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty. Cannot export or visualize.")

    ext = os.path.splitext(output_file)[1].lower()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if ext == ".html":
        net = Network(height='750px', width='100%', directed=True, notebook=False)
        net.barnes_hut()

        # Add nodes with full labeling and coloring
        for node, data in G.nodes(data=True):
            node_type = data.get("node_type", "default")
            label = data.get("name") or data.get("value") or node

            color = {
                "header": "blue",
                "column": "green",
                "row": "orange",
                "cell": "crimson"
            }.get(node_type, "gray")

            net.add_node(
                node,
                label=str(label),
                title=f"{node_type.upper()}: {label}",
                color=color
            )

        # Add edges with types as labels
        for u, v, d in G.edges(data=True):
            edge_type = d.get("type", "relation")
            net.add_edge(u, v, label=edge_type, title=edge_type)

        net.write_html(output_file)
        print(f"[✔] HTML visualization saved: {output_file}")
        webbrowser.open(f"file://{os.path.realpath(output_file)}")

    elif ext == ".json":
        data = nx.node_link_data(G)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[✔] JSON graph saved: {output_file}")

    elif ext == ".graphml":
        nx.write_graphml(G, output_file)
        print(f"[✔] GraphML saved: {output_file}")

    elif ext == ".edgelist":
        nx.write_edgelist(G, output_file, data=["type"])
        print(f"[✔] Edge list saved: {output_file}")

    elif ext == ".pt":
        # Step 1: Get all possible node attribute keys
        all_node_attrs = set()
        for _, data in G.nodes(data=True):
            all_node_attrs.update(data.keys())

        # Step 2: Pad all nodes with missing keys
        for node, data in G.nodes(data=True):
            for key in all_node_attrs:
                if key not in data:
                    data[key] = None

        pyg_graph = from_networkx(G)
        torch.save(pyg_graph, output_file)
        print(f"[✔] PyTorch Geometric graph saved: {output_file}")

    else:
        raise ValueError(f"[✖] Unsupported export format: {ext}")
