from collections import defaultdict
import numpy as np
import random
from cos import compute_and_store_neighbors
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def connected_components(list_of_nodes):
    '''
    takes in your graph’s list of nodes, and returns a list of lists – each inner list contains all nodes with a common label

    Inputs: list of nodes in graph
    Outputs: returns 2D array of nodes grouped by label

    '''

    common_labels = defaultdict(list)

    for node in list_of_nodes:
        common_labels[node.label].append(node)

    return np.array(list(common_labels.values()), dtype = object)


def propagate_label(node, node_neighbors, adj_matrix):
    '''
    takes in a node, the node’s neighbors, and the adjacency matrix. It should update that node’s label based on the weights of its neighbor’s labels

    Inputs: node, list of node's neighbors, adjacency matrix
    Outputs: updates node's label by adding up weights of neighbor labels
    '''
    weights = defaultdict(float)

    for neighboring_node in node_neighbors:
        weights[neighboring_node.label] += adj_matrix[node.id][neighboring_node.id]

    if weights:
        best_label = max(weights.items(), key=lambda x: x[1])[0]
        node.label = best_label

def whispers(nodes, adj_matrix, threshold, iterations = 50):
    num_components = []

    for node in nodes:
        compute_and_store_neighbors(node, nodes, adj_matrix, threshold)
    
    for iteration in range(iterations):
        node = random.choice(nodes)
        neighbor_tuples = node.get_neighbors()

        neighbors_node = []
        for i in range(len(neighbor_tuples)):
            neighbors_node.append(neighbor_tuples[i][0])

        propagate_label(node, neighbors_node, adj_matrix)
        components = connected_components(nodes)
        num_components.append(len(components))
    
def plot_graph(graph, adj):
    """ Use the package networkx to produce a diagrammatic plot of the graph, with
    the nodes in the graph colored according to their current labels.
    Note that only 20 unique colors are available for the current color map,
    so common colors across nodes may be coincidental.
    Parameters
    ----------
    graph : Tuple[Node, ...]
        The graph to plot. This is simple a tuple of the nodes in the graph.
        Each element should be an instance of the `Node`-class.

    adj : numpy.ndarray, shape=(N, N)
        The adjacency-matrix for the graph. Nonzero entries indicate
        the presence of edges.

    Returns
    -------
    Tuple[matplotlib.fig.Fig, matplotlib.axis.Axes]
        The figure and axes for the plot."""

    g = nx.Graph()
    for n, node in enumerate(graph):
        g.add_node(n)

    # construct a network-x graph from the adjacency matrix: a non-zero entry at adj[i, j]
    # indicates that an egde is present between Node-i and Node-j. Because the edges are
    # undirected, the adjacency matrix must be symmetric, thus we only look ate the triangular
    # upper-half of the entries to avoid adding redundant nodes/edges
    g.add_edges_from(zip(*np.where(np.triu(adj) > 0)))

    # we want to visualize our graph of nodes and edges; to give the graph a spatial representation,
    # we treat each node as a point in 2D space, and edges like compressed springs. We simulate
    # all of these springs decompressing (relaxing) to naturally space out the nodes of the graph
    # this will hopefully give us a sensible (x, y) for each node, so that our graph is given
    # a reasonable visual depiction
    pos = nx.spring_layout(g)

    # make a mapping that maps: node-lab -> color, for each unique label in the graph
    color = list(iter(cm.tab20b(np.linspace(0, 1, len(set(i.label for i in graph))))))
    color_map = dict(zip(sorted(set(i.label for i in graph)), color))
    colors = [color_map[i.label] for i in graph]  # the color for each node in the graph, according to the node's label

    # render the visualization of the graph, with the nodes colored based on their labels!
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(g, pos=pos, ax=ax, nodelist=range(len(graph)), node_color=colors)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=g.edges())
    return fig, ax

    

    

