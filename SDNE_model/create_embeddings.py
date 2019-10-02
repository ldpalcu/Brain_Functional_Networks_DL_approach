import numpy as np
import utils as us
import os
from numba import jit
from numba import njit

from core import SDNE

# %% Global variable
encode_dim = 8
nr_epochs = 10
batch_size = 10
path_to_read = "./dataset"
path_to_write = "./test"

folders = ["sum_weight_10","sum_weight_20","sum_weight_30","sum_weight_40","sum_weight_50","sum_weight_60","sum_weight_70",
           "sum_weight_80", "sum_weight_90", "sum_weight_100"]

states = []
format = 1

folder = "sum_weight_70"


# %%Utilities functions

def get_graph_embeddings_nodes(graph, nr_dims, nr_epochs, batch_size):
    model_clustered = SDNE(graph, encode_dim=nr_dims, alpha=2)
    model_clustered.fit(batch_size=batch_size, epochs=nr_epochs, verbose=1)

    return model_clustered.get_node_embedding()


# %% Reading data

graphs_dict = {}

# for folder in folders:
print(folder)
for state in states:
    graphs_path = os.path.join(path_to_read, folder)
    graphs_state_dict = us.read_graphs_from_directory(graphs_path, threshold=None, state=state, format=format)
    graphs_dict.update(graphs_state_dict)

# %% Embedding data

path_folder = os.path.join(path_to_write, folder)
if not os.path.exists(path_folder):
    os.mkdir(path_folder)
for graph_name, graph in graphs_dict.iteritems():
    print(graph_name)
    embedding = get_graph_embeddings_nodes(graph, encode_dim, nr_epochs, batch_size)
    graph_name, _ = graph_name.split(".")
    graph_path = os.path.join(path_folder, graph_name + ".csv")
    np.savetxt(graph_path, embedding, delimiter=",")
