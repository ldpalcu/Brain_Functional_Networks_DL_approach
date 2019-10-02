import utils as us
from FunctionalNetwork import FunctionalNetwork

# %% Global variables
nr_animals = 19
pre_processed_data_path = "pre_processed_data/format_state_animal/demo_graphs"
raw_data_path = "new_graphs_met2"
threshold = "sum_weight"
format = 1

# %% Create a tree directory format
# for i in range(0, 110, 10):
#     us.create_tree_directories(pre_processed_data_path, threshold + "_" + str(i), nr_animals, format)

# %% Read data
graphs = us.read_raw_data_from_directory(raw_data_path)
# graphs = us.read_raw_data_from_directory("raw_functional_networks")


# %% Create subgraphs
subgraphs = {}
for i in range(70, 110, 10):
    print(i)
    for graph_name, graph in graphs.items():
        graph.remove_edges_from(graph.selfloop_edges())
        fn = FunctionalNetwork(graph)
        cnt_pos, sum_pos, cnt_neg, sum_neg = fn.nr_and_sum_positive_and_negative_edges()
        subgraph = fn.create_subgraph_using_sum_edges(sum_pos, sum_neg, i, order_mode=True)
        subgraphs.update({graph_name: subgraph})
    us.write_graph_to_directory(pre_processed_data_path, threshold + "_" + str(i), subgraphs, format)
    break

# %% Write subgraphs to directory
print("final")
