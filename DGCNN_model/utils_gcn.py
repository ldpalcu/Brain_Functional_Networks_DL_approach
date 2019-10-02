"""
Functions used for processing data
"""

import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import re

states = []


def accept_file(filename):
    if "ValAtTimeOffset" not in filename:
        return False
    return True


def draw_graph(G):
    """Draw graph using matplotlib"""

    edge_labels = dict(((u, v), d["weight"]) for u, v, d in G.edges(data=True))
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos, node_size=500, with_labels=True)
    plt.show()


def read_raw_data_from_directory(path):
    """ Read the original data from a directory

    :param path: directory
    :return: original data as networkx graphs
    """
    graphs = {}
    for filename in sorted(os.listdir(path)):
        # accept only files ValAtTimeOffset
        if accept_file(filename):
            input_data = pd.read_csv(path + "/" + filename, header=None)
            G = nx.Graph(input_data.values)
            graphs.update({filename: G})
    return graphs


def write_graph_to_directory(path, threshold, graphs, format):
    """Write graphs into directory

    :param path:
    :param threshold:
    :param graphs:
    :param format:
    :return:
    """
    for key, value in graphs.iteritems():
        _, animal_id, state, trial, _, = key.split("-")
        file_name = str(animal_id) + "-" + state + "-" + trial + ".xml"
        if not format:
            new_file_path = path + "/" + str(threshold) + "/" + str(animal_id) + "/" + state
        else:
            new_file_path = path + "/" + str(threshold) + "/" + state

        if os.path.exists(new_file_path):
            nx.write_graphml(value, new_file_path + "/" + file_name)


def read_graphs_from_directory(path, threshold, state, format, animal_id=None):
    """Read graphs from directory
    :param path:
    :param threshold:
    :param animal_id:
    :param state:
    :param format:
    :return:
    """
    if not format:
        new_file_path = path + "/" + str(threshold) + "/" + str(animal_id) + "/" + state
    else:
        new_file_path = path + "/" + str(threshold) + "/" + state

    graphs = {}
    for filename in os.listdir(new_file_path):
        if "xml" in filename:
            graph = nx.read_graphml(new_file_path + "/" + filename, node_type=int)
            graphs.update({filename: graph})

    return graphs


def read_graph_from_directory(path, threshold, animal_id, state, format):
    """ Read a specific graph from a specific directory

    :param path:
    :param threshold:
    :param animal_id:
    :param state:
    :param format:
    :return:
    """
    if not format:
        new_file_path = path + "/" + str(threshold) + "/" + str(animal_id) + "/" + state
    else:
        new_file_path = path + "/" + str(threshold) + "/" + state

    for filename in os.listdir(new_file_path):
        if re.match(str(animal_id) + "-" + state + "-[0-9]+\.xml", filename):
            return nx.read_graphml(new_file_path + "/" + filename)


def create_tree_directories(path, threshold, nr_animals, format):
    """Create a tree directory. There are two formats: 0 means that the files are organized per threshold, then
    per animal and then per state, 1 means that the files are organized per threshold(let's say condition) and
    then per state.

    :param path:
    :param threshold:
    :param nr_animals:
    :param format:
    :return:
    """
    # create a subdirectory based on given threshold
    new_dir_threshold_path = path + "/" + str(threshold)
    if os.path.exists(new_dir_threshold_path):
        # delete existing directory because this means that you want to replace the old data
        shutil.rmtree(new_dir_threshold_path)

    os.mkdir(new_dir_threshold_path)

    # create a subdirectory for an animal if format animal-condition
    if not format:
        for animal_id in range(1, nr_animals):
            new_dir_animal_path = ""
            if not format:
                new_dir_animal_path = new_dir_threshold_path + "/" + str(animal_id)
                os.mkdir(new_dir_animal_path)

            # create a subdirectory for every animal state
            for state in states:
                new_dir_state_path = new_dir_animal_path + "/" + state
                os.mkdir(new_dir_state_path)
    else:
        for state in states:
            new_dir_state_path = new_dir_threshold_path + "/" + state
            os.mkdir(new_dir_state_path)

    if not format:
        return new_dir_animal_path
    else:
        return new_dir_threshold_path

