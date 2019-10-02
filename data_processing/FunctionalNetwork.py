"""
Representing the functional network of the brain
"""

import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections
import numpy as np


class FunctionalNetwork:

    def __init__(self, G):
        self.graph = G

    def draw_graph(self):
        edge_labels = dict(((u, v), d["weight"]) for u, v, d in self.graph.edges(data=True))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        nx.draw(self.graph, pos, node_size=500, with_labels=True)
        plt.show()

    def draw_graph_using_cytoscape(self, name):
        nx.write_graphml(self.graph, name)

    def create_subgraph(self, threshold_pos, threshold_neg):
        """Create subgraph by cutting edges under a given threshold

        :param threshold_pos:
        :param threshold_neg:
        :return:
        """
        subgraph_edges = []
        for u, v, attr in self.graph.edges(data=True):
            if attr['weight'] >= 0:
                if attr['weight'] >= threshold_pos:
                    subgraph_edges.append((u, v))
            else:
                if abs(attr['weight']) >= threshold_neg:
                    subgraph_edges.append((u, v))

        return self.graph.edge_subgraph(subgraph_edges)

    def get_number_of_edges(self):
        return self.graph.number_of_edges()

    def get_average_degree(self):
        return self.graph.number_of_nodes() / self.graph.number_of_edges()

    def draw_degree_histogram(self, name):
        degree_sequence = sorted([d for n, d in self.graph.degree()], reverse=True)
        degree_count = collections.Counter(degree_sequence)
        deg, cnt = zip(*degree_count.items())

        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color='b')

        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d for d in deg])
        ax.set_xticklabels(deg)

        plt.savefig("first_trial_distribution/" + name + ".png")

        plt.show()

    def nr_connected_components(self):
        return nx.number_connected_components(self.graph)

    def nr_isolated_nodes(self):
        return nx.number_of_isolates(self.graph)

    def draw_edge_weight_distribution(self, name):
        keys_positive = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                         0.8, 0.9, 1]

        edges_positive = {key: 0 for key in keys_positive}

        edges_positive[0] = {"neg": 0, "pos": 0}

        for u, v, attr in self.graph.edges(data=True):
            key = round(int(attr['weight'] * 10) * 0.1, 1)
            if key == 0:
                if attr['weight'] >= 0:
                    edges_positive[key]["pos"] = edges_positive[key]["pos"] + 1
                else:
                    edges_positive[key]["neg"] = edges_positive[key]["neg"] + 1
            else:
                edges_positive[key] = edges_positive[key] + 1

        # for key, value in edges_positive.iteritems():
        #     edges_positive[key] = value / 2

        lists = sorted(edges_positive.items())

        x, y = zip(*lists)
        x = list(x)
        y = list(y)

        plt.figure(figsize=(10, 7))

        plt.bar(x[0:10], y[0:10], width=-0.08, color='k', align="edge")
        plt.bar(x[10], y[10]["neg"], width=-0.08, color='k', align="edge")
        plt.bar(x[10], y[10]["pos"], width=0.08, color='r', align="edge")
        plt.bar(x[11:20], y[11:20], width=0.08, color='r', align="edge")
        plt.bar(x[20], y[20], width=0.05, color='r', align='center')
        plt.xticks([s for s in x])

        plt.title("Edge weight distribution")
        plt.ylabel("Count")
        plt.xlabel("Edge weight")


        plt.savefig(name)
        # plt.show()

    def nr_and_sum_positive_and_negative_edges(self):
        cnt_pos = 0
        sum_pos = 0
        cnt_neg = 0
        sum_neg = 0
        for u, v, attr in self.graph.edges(data=True):
            if attr['weight'] >= 0:
                cnt_pos = cnt_pos + 1
                sum_pos = sum_pos + attr['weight']
            else:
                cnt_neg = cnt_neg + 1
                sum_neg = sum_neg + attr['weight']

        return cnt_pos, sum_pos, cnt_neg, sum_neg

    def create_statistics(self, name):

        # TODO find connected components aka isolated nodes
        connected_components = nx.isolates(self.graph)
        nr_connected_components = nx.number_connected_components(self.graph)
        cnt_pos, sum_pos, cnt_neg, sum_neg = self.nr_and_sum_positive_and_negative_edges()

        graph_data = {"name": name, "nr_pos_edges": cnt_pos, "nr_neg_edges": cnt_neg, "sum_pos_edges": sum_pos,
                      "sum_neg_edges": sum_neg, "mean_pos_edges": 0 if cnt_pos == 0 else sum_pos / cnt_pos,
                      "mean_neg_edges": 0 if cnt_neg == 0 else sum_neg / cnt_neg,
                      "nr_connected_components": nr_connected_components}

        return graph_data

    def create_subgraph_using_sum_edges(self, sum_pos, sum_neg, threshold, order_mode):
        """Create a subgraph using sum of edges method: keeping those edges whose sum is under a given threshold.

        :param sum_pos:
        :param sum_neg:
        :param threshold:
        :return:
        """

        subgraph = self.graph.copy()

        # subgraph.remove_edges_from(self.graph.selfloop_edges())

        if order_mode:
            sorted_edges_pos = sorted(subgraph.edges(data=True), key=lambda x: -x[2]['weight'])
            sorted_edges_neg = sorted(subgraph.edges(data=True), key=lambda x: x[2]['weight'])
        else:
            sorted_edges_pos = sorted(subgraph.edges(data=True), key=lambda x: x[2]['weight'])
            sorted_edges_neg = sorted(subgraph.edges(data=True), key=lambda x: -x[2]['weight'])

        target_sum_pos = float(threshold) / 100 * sum_pos
        target_sum_neg = float(threshold) / 100 * abs(sum_neg)

        partial_sum_pos = 0
        partial_sum_neg = 0

        flag_break = False

        for u, v, attr in sorted_edges_pos:
            if attr['weight'] >= 0:
                if not flag_break:
                    if (attr['weight'] + partial_sum_pos) > target_sum_pos:
                        flag_break = True
                        if threshold == 100:
                            partial_sum_pos += attr['weight']
                        else:
                            subgraph.remove_edge(u, v)
                    else:
                        partial_sum_pos += attr['weight']
                else:
                    subgraph.remove_edge(u, v)

        flag_break = False

        for u, v, attr in sorted_edges_neg:
            if attr['weight'] < 0:
                if not flag_break:
                    if (abs(attr['weight']) + partial_sum_neg) > target_sum_neg:
                        flag_break = True
                        if threshold == 100:
                            partial_sum_neg += abs(attr['weight'])
                        else:
                            subgraph.remove_edge(u, v)
                    else:
                        partial_sum_neg += abs(attr['weight'])
                else:
                    subgraph.remove_edge(u, v)

        return subgraph
