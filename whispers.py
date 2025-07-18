from collections import defaultdict
import numpy as np

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

def whispers(nodes, adj_matrix):
    for i in enumerate(nodes):
        node.label = i