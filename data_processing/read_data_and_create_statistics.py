import utils as us

states = ["Control", "EtOH"] #"Naltrexone", "Abstinence"]
path_read = "pre_processed_data/format_state_animal/"
format = 1
raw_data_path = "synthetic_percentage_graphs/02P500Graphs"
path_to_write_statistics = "statistics/"
path_to_write_edge_weight_distribution = "edge_weight_distribution/" + "subgraphs_sum_weight/" + \
                                         "subgraphs_percentage_02/sum_weight" + "_" + "70"
animal_id = 1
original_data = False


# read original data and create statistics for it
if original_data:
    graphs = us.read_raw_data_from_directory(raw_data_path)
    # us.persist_statistics(graphs, path_to_write_statistics)
    us.persist_edge_weight_distribution(graphs,  path_to_write_edge_weight_distribution)
else:
    # read subgraphs
    for i in range(70, 110, 10):
        print(i)
        threshold = "new_generated_data_percentage_high/02P500Graphs/sum_weight" + "_" + str(i)
        # graph_statistics = "sum_weight_" + str(i) + "_" + "high_edge_values.csv"
        # new_path_to_write_statistics = path_to_write_statistics + graph_statistics
        new_path_to_write_edge_weight_distribution = path_to_write_edge_weight_distribution
        # graphs_statistics = {}
        for state in states:
            graphs = us.read_graphs_from_directory(path_read, threshold, animal_id, state, format)
            us.persist_edge_weight_distribution(graphs, new_path_to_write_edge_weight_distribution)
            # graphs_statistics.update(graphs)
        # us.persist_statistics(graphs_statistics, new_path_to_write_statistics)
        break

print("final")
