# graph_builder/visualize_graph.py

import networkx as nx
import plotly.graph_objects as go

def visualize_graph(G: nx.Graph, title="StructRelGraph"):
    pos = nx.spring_layout(G, seed=42, k=0.9)  # spring layout for clarity

    # Node type → color map
    color_map = {
        'header': 'blue',
        'column': 'green',
        'row': 'orange',
        'cell': 'crimson',
        'default': 'gray'
    }

    # Build traces for each node type
    node_traces = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('node_type', 'default')
        label = data.get("name") or data.get("value") or node

        if node_type not in node_traces:
            node_traces[node_type] = {
                'x': [],
                'y': [],
                'text': [],
                'marker': dict(size=14, color=color_map.get(node_type, 'gray'), line=dict(width=2)),
                'name': node_type
            }

        x, y = pos[node]
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)
        node_traces[node_type]['text'].append(f"<b>{node_type.upper()}</b>: {label}")

    # Draw all edges
    edge_x = []
    edge_y = []
    for u, v, _ in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.5, color='#999'),
        hoverinfo='none',
        mode='lines'
    )

    # Final combined figure
    fig = go.Figure()

    fig.add_trace(edge_trace)

    for node_type, trace_data in node_traces.items():
        fig.add_trace(go.Scatter(
            x=trace_data['x'],
            y=trace_data['y'],
            mode='markers+text',
            text=trace_data['text'],
            textposition='top center',
            hoverinfo='text',
            marker=trace_data['marker'],
            name=node_type
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        annotations=[
            dict(
                text="Drag nodes to explore structure",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    fig.show()
