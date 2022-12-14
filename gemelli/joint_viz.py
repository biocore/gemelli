import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from gemelli.rpca import joint_rpca


def create_graph(correlation_table,
                 feature_map,
                 features_use=None,
                 pos_corr_thresh=0.5,
                 neg_corr_thresh=-0.5):
    if features_use is not None:
        correlation_table = correlation_table.loc[features_use, features_use]

    idx = correlation_table.index.values
    G = nx.from_numpy_matrix(correlation_table.values)
    G = nx.relabel_nodes(G, lambda x: idx[x])
    for _id in idx:
        G.remove_edge(_id, _id)
    nx.set_node_attributes(G, feature_map, 'modality')

    edges_to_keep = []
    for node1, node2, attr in G.edges(data=True):
        weight = attr['weight']
        if weight > pos_corr_thresh or weight < neg_corr_thresh:
            edges_to_keep.append((node1, node2))

    G = G.edge_subgraph(edges_to_keep)
    return G


def visualize_graph(G, feature_map):
    labels = nx.get_node_attributes(G, 'modality')

    modalities = list(set(feature_map.values()))
    num_modalities = len(modalities)

    edge_weights = []
    edge_colors = []
    for u, v, attr in G.edges(data=True):
        weight = attr['weight']
        if weight > 0:
            ec = "blue"
        elif weight < 0:
            ec = "red"
        else:
            ec = "gray"
        edge_colors.append(ec)
        edge_weights.append(np.abs(weight)*0.5)

    palette = dict(zip(modalities, sns.color_palette("tab10", num_modalities)))
    node_colors = [palette[G.nodes[node]["modality"]] for node in G.nodes]
    labels = nx.get_node_attributes(G, "modality")

    fig, ax = plt.subplots(1, 1)

    nx.draw_networkx(
        G,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_weights,
        ax=ax,
        with_labels=False
    )

    handles = []
    pos_line = Line2D([0], [0], label="positive", color="blue")
    neg_line = Line2D([0], [0], label="negative", color="red")
    handles.extend([pos_line, neg_line])

    for modality, color in palette.items():
        p = Line2D([0], [0], mfc=color, label=modality, markersize=10,
                   marker="o", mew=0, linewidth=0)
        handles.append(p)

    ax.legend(handles=handles)
    return ax
