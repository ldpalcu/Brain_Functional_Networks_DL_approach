import utils_gcn as us
import networkx as nx
import collections

path_to_read = ""
threshold = ""
states = []
format = 1
tag = 0


def create_labels(state):
    switcher = {}

    return switcher.get(state, lambda: "Invalid state")


def read_graphs():
    print('loading data')
    all_graphs = collections.OrderedDict()

    for state in states:
        all_graphs.update(
            us.read_graphs_from_directory(path=path_to_read,
                                          threshold=threshold,
                                          state=state,
                                          format=format))

    return all_graphs


def filter_graphs(all_graphs, id_examples):
    filtered_graphs = collections.OrderedDict()
    for graph_name, graph in all_graphs.items():
        nr_example, _, _ = graph_name.split("-")
        if int(nr_example) in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
            filtered_graphs.update({graph_name: graph})

    return filtered_graphs


def filter_nodes_graphs(all_graphs, id_nodes):
    filtered_graphs = collections.OrderedDict()
    for graph_name, graph in all_graphs.items():
        for id_node in id_nodes:
            edges = graph.edges(id_node)
            graph.remove_edges_from(edges)
        filtered_graphs.update({graph_name: graph})

    return filtered_graphs


def transform_xml_to_txt(all_graphs):
    with open("", "w") as f:
        nr_graphs = len(all_graphs)
        f.write(str(nr_graphs) + "\n")
        for graph_name, graph in all_graphs.items():
            print(graph_name)
            # nr_example, state, _ = graph_name.split("-")
            nr_example, state = graph_name.split("-")
            state, _ = state.split(".")
            label = create_labels(state)
            print(label)

            f.write("{} {}\n".format(str(graph.number_of_nodes()), str(label)))
            nodes = sorted(list(graph.nodes()))
            for n in nodes:
                neighbors = sorted(list(graph.neighbors(n)))
                if not neighbors:
                    f.write("{} {} ".format(str(tag), str(0)))
                else:
                    f.write("{} {} ".format(str(tag), str(len(neighbors))))
                    for neighbor in neighbors:
                        f.write("{} ".format(str(neighbor)))
                f.write("\n")


def write_edges(all_graphs):
    with open("edges.txt", "w") as e:
        for graph_name, graph in all_graphs.items():
            print(graph_name)

            graph_edges = list(graph.edges(data='weight'))

            e.write(str(len(graph_edges)) + "\n")
            e.write("\n".join("{} {} {}".format(x[0], x[1], x[2])
                              for x in graph_edges))
            e.write("\n")


def write_features(all_graphs):
    with open("", "w") as f:

        nr_graphs = len(all_graphs)
        f.write(str(nr_graphs) + "\n")

        for graph_name, graph in all_graphs.items():
            print(graph_name)
            nr_example, state, _ = graph_name.split("-")
            label = create_labels(state)
            print(label)

            e_centrality = nx.eigenvector_centrality_numpy(graph)
            bet_centrality = nx.betweenness_centrality(graph, k=30)
            # cf_closeness_centrality = nx.current_flow_closeness_centrality(graph)
            # cf_bet_centrality = nx.current_flow_betweenness_centrality(graph)
            load_centrality = nx.load_centrality(graph)
            clustering_coeff = nx.clustering(graph)
            square_clustering_coeff = nx.square_clustering(graph)

            f.write("{} {}\n".format(str(graph.number_of_nodes()), str(label)))
            nodes = sorted(list(graph.nodes()))
            for n in nodes:
                neighbors = sorted(list(graph.neighbors(n)))
                if not neighbors:
                    f.write("{} {} ".format(str(tag), str(0)))
                else:
                    f.write("{} {} ".format(str(tag), str(len(neighbors))))
                    for neighbor in neighbors:
                        f.write("{} ".format(str(neighbor)))
                    cc = nx.closeness_centrality(graph, u=n)
                # degree
                f.write("{} ".format(graph.degree(n)))
                # eigenvector centrality
                f.write("{} ".format(e_centrality[n]))
                # betweeness centrality
                f.write("{} ".format(bet_centrality[n]))
                # closeness centrality
                f.write("{} ".format(cc))
                # current flow closeness centrality
                f.write("{} ".format(cf_closeness_centrality[n]))
                # current flow betweeness centrality
                f.write("{} ".format(cf_bet_centrality[n]))
                # load centrality
                f.write("{} ".format(load_centrality[n]))
                # clustering coefficient
                f.write("{} ".format(clustering_coeff[n]))
                # square clustering coefficient
                f.write("{} ".format(square_clustering_coeff[n]))
                f.write("\n")


def create_train_splits(all_graphs, example_split, idx_nr):
    with open("test_idx-" + str(idx_nr) + ".txt",
              "w") as g_test, open("train_idx-" + str(idx_nr) + ".txt",
                                   "w") as g_train:
        k = 0
        for graph_name, graph in all_graphs.items():
            print(graph_name)
            # nr_example, state, _ = graph_name.split("-")
            nr_example, state = graph_name.split("-")
            state, _ = state.split(".")
            label = create_labels(state)
            print(label)

            if int(nr_example) in example_split:
                g_test.write("{}\n".format(str(k)))
            else:
                g_train.write("{}\n".format(str(k)))
            k = k + 1


examples_split = [
    range(1, 51, 1),
    range(51, 101, 1),
    range(101, 151, 1),
    range(151, 201, 1),
    range(201, 251, 1),
    range(1, 101, 2),
    range(101, 201, 2),
    range(151, 251, 2),
    range(51, 151, 2),
    range(51, 76, 1) + range(201, 226, 1)
]
examples_split_2 = [[1, 3], [5, 7], [9, 11], [13, 15], [15, 17], [11, 13],
                    [7, 9], [3, 5], [1, 17], [7, 11]]
examples_split_3 = [[2, 4], [6, 8], [10, 12], [14, 16], [16, 18], [12, 14],
                    [8, 10], [4, 6], [2, 18], [6, 12]]
filtered_examples_id = [[1]]


def pipeline_processing_data():
    all_graphs = read_graphs()
    print(len(all_graphs))
    write_edges(all_graphs)
    transform_xml_to_txt(all_graphs)
    idx_nr = 1
    for example_split in examples_split:
        create_train_splits(all_graphs, example_split, idx_nr)
        idx_nr += 1


pipeline_processing_data()
