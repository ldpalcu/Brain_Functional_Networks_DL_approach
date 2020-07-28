import utils_gcn as us
import networkx as nx
import collections
import random
import numpy as np
import os

path_to_read = "pre_processed_data/data_knn/fourth_window"
path_to_write = "fourth_window_knn_graphs.txt"
threshold = ""
states = ["UNSEEN", "SEEN"]
format = 1
tag = 0


def create_labels(state):
    switcher = {states[0]: 0, states[1]: 1}

    return switcher.get(state, lambda: "Invalid state")


def read_graphs():
    print('loading data')
    all_graphs = collections.OrderedDict()
    index = 0
    for patient_number in range(1, 12):
        print(patient_number)
        patient_folder = "patient_{}".format(patient_number)
        new_file_path = os.path.join(path_to_read, patient_folder)
        for filename in os.listdir(new_file_path):
            if "UNCERTAIN" in filename:
                continue
            graph = nx.read_graphml(new_file_path + "/" + filename, node_type=int)
            nr_example, state = filename.split("_")
            state, _ = state.split(".")
            state = create_labels(state)
            all_graphs.update({index: (graph, state, patient_number)})
            index += 1
        print(len(all_graphs))

    # for state in states:
    #     all_graphs.update(
    #         us.read_graphs_from_directory(path=path_to_read,
    #                                       threshold=threshold,
    #                                       state=state,
    #                                       format=format))

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
    with open(path_to_write, "w") as f:
        nr_graphs = len(all_graphs)
        f.write(str(nr_graphs) + "\n")
        for index, g_s in all_graphs.items():
            graph = g_s[0]
            label = g_s[1]
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
        # for graph_name, graph in all_graphs.items():
        #     print(graph_name)
        #     # nr_example, state, _ = graph_name.split("-")
        #     nr_example, state = graph_name.split("_")
        #     state, _ = state.split(".")
        #     label = create_labels(state)
        #     print(label)

        #     f.write("{} {}\n".format(str(graph.number_of_nodes()), str(label)))
        #     nodes = sorted(list(graph.nodes()))
        #     for n in nodes:
        #         neighbors = sorted(list(graph.neighbors(n)))
        #         if not neighbors:
        #             f.write("{} {} ".format(str(tag), str(0)))
        #         else:
        #             f.write("{} {} ".format(str(tag), str(len(neighbors))))
        #             for neighbor in neighbors:
        #                 f.write("{} ".format(str(neighbor)))
        #         f.write("\n")


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


# def create_train_splits(all_graphs, example_split_1, example_split_2, idx_nr):
#     with open("test_idx-" + str(idx_nr) + ".txt",
#               "w") as g_test, open("train_idx-" + str(idx_nr) + ".txt",
#                                    "w") as g_train:
#         k = 0
#         for graph_name, graph in all_graphs.items():
#             # print(graph_name)
#             # nr_example, state, _ = graph_name.split("-")
#             nr_example, state = graph_name.split("-")
#             state, _ = state.split(".")
#             label = create_labels(state)
#             print(label)

#             if state == states[0]:
#                 if int(nr_example) in example_split_1:
#                     g_test.write("{}\n".format(str(k)))
#                 else:
#                     g_train.write("{}\n".format(str(k)))
#             elif state == states[1]:
#                 if int(nr_example) in example_split_2:
#                     g_test.write("{}\n".format(str(k)))
#                 else:
#                     g_train.write("{}\n".format(str(k)))
#             k = k + 1

def create_train_splits(all_graphs, reference_patient_number, idx_nr):
    with open("test_idx-" + str(idx_nr) + ".txt",
              "w") as g_test, open("train_idx-" + str(idx_nr) + ".txt",
                                   "w") as g_train:
        k = 0
        for index, g_s_p in all_graphs.items():
            patient_number = g_s_p[2]
            if patient_number == reference_patient_number:
                g_test.write("{}\n".format(str(index)))
            else:
                g_train.write("{}\n".format(str(index)))
        # for graph_name, graph in all_graphs.items():
        #     # print(graph_name)
        #     # nr_example, state, _ = graph_name.split("-")
        #     nr_example, state = graph_name.split("-")
        #     state, _ = state.split(".")
        #     label = create_labels(state)
        #     print(label)

        #     if state == states[0]:
        #         if int(nr_example) in example_split_1:
        #             g_test.write("{}\n".format(str(k)))
        #         else:
        #             g_train.write("{}\n".format(str(k)))
        #     elif state == states[1]:
        #         if int(nr_example) in example_split_2:
        #             g_test.write("{}\n".format(str(k)))
        #         else:
        #             g_train.write("{}\n".format(str(k)))
        #     k = k + 1

def create_examples_split():
    elems = []
    elems.append(
        random.sample(list(range(0, 50, 1)) + list(range(250, 300, 1)), 100))
    elems.append(
        random.sample(list(range(50, 100, 1)) + list(range(300, 350, 1)), 100))
    elems.append(
        random.sample(
            list(range(100, 150, 1)) + list(range(350, 400, 1)), 100))
    elems.append(
        random.sample(
            list(range(150, 200, 1)) + list(range(400, 450, 1)), 100))
    elems.append(
        random.sample(
            list(range(200, 250, 1)) + list(range(450, 500, 1)), 100))
    elems.append(
        random.sample(list(range(0, 100, 2)) + list(range(250, 350, 2)), 100))
    elems.append(
        random.sample(
            list(range(100, 200, 2)) + list(range(350, 450, 2)), 100))
    elems.append(
        random.sample(list(range(1, 101, 2)) + list(range(251, 351, 2)), 100))
    elems.append(
        random.sample(
            list(range(101, 201, 2)) + list(range(351, 451, 2)), 100))
    elems.append(
        random.sample(
            list(range(150, 250, 2)) + list(range(400, 500, 2)), 100))

    # for k in range(0, len(elems)):
    #     np.random.shuffle(elems[k])

    return elems


def create_examples_split_imbalanced():
    elems_1 = []
    elems_1.append(random.sample(list(range(0, 90, 1)), 90))
    elems_1.append(random.sample(list(range(90, 180, 1)), 90))
    elems_1.append(random.sample(list(range(180, 270, 1)), 90))
    elems_1.append(random.sample(list(range(270, 360, 1)), 90))
    elems_1.append(random.sample(list(range(360, 450, 1)), 90))
    elems_1.append(random.sample(list(range(0, 180, 2)), 90))
    elems_1.append(random.sample(list(range(180, 360, 2)), 90))
    elems_1.append(random.sample(list(range(91, 271, 2)), 90))
    elems_1.append(random.sample(list(range(181, 361, 2)), 90))
    elems_1.append(random.sample(list(range(270, 450, 2)), 90))

    elems_2 = []
    elems_2.append(random.sample(list(range(450, 460, 1)), 10))
    elems_2.append(random.sample(list(range(460, 470, 1)), 10))
    elems_2.append(random.sample(list(range(470, 480, 1)), 10))
    elems_2.append(random.sample(list(range(480, 490, 1)), 10))
    elems_2.append(random.sample(list(range(490, 500, 1)), 10))
    elems_2.append(random.sample(list(range(450, 470, 2)), 10))
    elems_2.append(random.sample(list(range(470, 490, 2)), 10))
    elems_2.append(random.sample(list(range(451, 471, 2)), 10))
    elems_2.append(random.sample(list(range(471, 491, 2)), 10))
    elems_2.append(random.sample(list(range(480, 500, 2)), 10))

    return elems_1, elems_2


def pipeline_processing_data():
    all_graphs = read_graphs()
    print(len(all_graphs))
    #write_edges(all_graphs)
    transform_xml_to_txt(all_graphs)
    idx_nr = 1
    for patient_nr in range(1, 11):
        create_train_splits(all_graphs, patient_nr, idx_nr)
        idx_nr += 1
    # created_examples_split = create_examples_split()
    # examples_split_1, examples_split_2 = create_examples_split_imbalanced()
    # example_split = []
    # example_split.append(examples_split_1)
    # example_split.append(examples_split_2)
    # for k in range(0, 10):
    #     print(idx_nr)
    #     create_train_splits(all_graphs, example_split[0][k],
    #                         example_split[1][k], idx_nr)
    #     idx_nr += 1


pipeline_processing_data()
