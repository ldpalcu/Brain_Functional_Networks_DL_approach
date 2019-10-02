import pickle
from random import randint
from random import uniform

import networkx as nx
import numpy as np
import pandas as pd

from numpy import genfromtxt
import matplotlib.pyplot as plt


def generate_joint_degree_sequence(nr_nodes, nr_triangles):
    degree_list = []
    for k in range(0, nr_nodes):
        degree_list.append((randint(0, nr_nodes - 1), randint(0,
                                                              nr_nodes - 1)))

    nr_zero_nodes = int((1 - nr_triangles) * nr_nodes)

    for k in range(0, nr_zero_nodes):
        idx_node = randint(0, nr_nodes - 1)
        degree_list[idx_node] = (degree_list[idx_node][0], 0)

    sum_degree = sum([x for x, _ in degree_list])

    sum_triangle = sum([x for _, x in degree_list])

    if sum_triangle % 3 != 0:
        new_elem_triangle = (int(sum_triangle / 3) + 1) * 3 - sum_triangle
        degree_list.append((0, new_elem_triangle))

    if sum_degree % 2 != 0:
        first_elem = degree_list[0]
        del degree_list[0]
        degree_list.append((first_elem[0] + 1, first_elem[1]))

    return degree_list


def generate_graph(nr_nodes, nr_triangles):
    degree_list_graph = generate_joint_degree_sequence(nr_nodes, nr_triangles)

    # generate random clustered graphs
    g = nx.random_clustered_graph(degree_list_graph)

    # delete self-loops and parallel edges
    g = nx.Graph(g)
    g.remove_edges_from(g.selfloop_edges())

    m_after_cutting_edges = g.number_of_edges()

    # generate random weights between 0 and 1
    weights = [
        round(uniform(0, 1), 4) for r in range(0, m_after_cutting_edges)
    ]
    uw_edges = [e for e in g.edges]

    w_edges = [(uw_edges[i][0], uw_edges[i][1], weights[i])
               for i in range(0, m_after_cutting_edges)]

    g = nx.Graph()

    g.add_weighted_edges_from(w_edges)

    return g


def generate_dataset():
    for i in range(0, 100):
        print(i)
        with open("graph_files_train/graph_clustered_" + str(i) + ".csv",
                  "wb") as f:
            G_clustered = generate_graph(80, 0.9)
            np.savetxt(f,
                       nx.to_numpy_array(G_clustered).astype(float),
                       fmt="%.0f")
        with open("graph_files_train/graph_less_clustered_" + str(i) + ".csv",
                  "wb") as g:
            G_less_clustered = generate_graph(80, 0.1)
            np.savetxt(g,
                       nx.to_numpy_array(G_less_clustered).astype(float),
                       fmt="%.0f")


generate_dataset()
