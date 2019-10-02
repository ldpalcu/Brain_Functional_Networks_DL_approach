from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

nr_nodes = 85

path_to_files = "./"
path_to_save = "./"

dataset = ""

int_to_state = {
}

states = []

# create an average heatmap for every state
mean_graphs = {}

for i in range(1, 2):
    file = "cur_message_layer_" + dataset + str(i) + ".txt"
    label_file = "labels_" + dataset + str(i) + ".txt"
    path_to_read_feature_maps = path_to_files + file
    path_to_read_labels = path_to_files + label_file

    features_map = np.loadtxt(path_to_read_feature_maps)
    labels = np.loadtxt(path_to_read_labels, dtype=int)

    graphs_per_state = {
        
    }
    label = 0
    for i in range(0, len(features_map), nr_nodes):
        graph = features_map[i: i + nr_nodes, :]
        graphs_per_state[int_to_state[labels[label]]].append(graph)
        label += 1

    for state in states:
        graphs = graphs_per_state[state]
        sum_graph = sum(graphs)
        mean_graph = sum_graph / len(graphs)
        mean_graphs[state].append(mean_graph)

for state in states:
    graphs = mean_graphs[state]
    sum_graph = sum(graphs)
    mean_graph = sum_graph / len(graphs)
    heatmap_state = sns.heatmap(mean_graph, xticklabels=1)
    fig = heatmap_state.get_figure()
    fig.savefig(path_to_save + state + "_" + dataset + ".png")
    fig.clf()
