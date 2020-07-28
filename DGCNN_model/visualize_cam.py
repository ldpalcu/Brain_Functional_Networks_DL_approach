import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def create_heatmap_cam_2d(cam, path_to_save, nodes_indexes, confidence_score):
    f, ax = plt.subplots(figsize=(15, 15))
    plt.figtext(x=0.13,
                y=0.90,
                s="Confidence score: {}".format(confidence_score),
                fontsize=15,
                fontname="sans-serif")
    heatmap_state = sns.heatmap(cam,
                                cmap="jet",
                                yticklabels=nodes_indexes,
                                ax=ax,
                                linewidths=.1,
                                linecolor='black')
    fig = heatmap_state.get_figure()
    fig.savefig(path_to_save)
    fig.clf()


def create_heatmap_cam_1d(cam, path_to_save, nodes_indexes, confidence_score):
    fig = plt.figure(figsize=(30, 10))
    plt.yticks([])
    plt.xticks(range(0, len(cam)), nodes_indexes)
    plt.figtext(x=0.13,
                y=0.54,
                s="Confidence score: {}".format(confidence_score),
                fontsize=15,
                fontname="sans-serif")
    y = 0.40
    for i in range(0, 10):
        plt.figtext(x=0.13,
                    y=y,
                    s="{} {}".format(i, nodes_indexes[i]),
                    fontsize=13,
                    fontname="sans-serif")
        y -= 0.03
    y = 0.40
    for i in range(10, 20):
        plt.figtext(x=0.30,
                    y=y,
                    s="{} {}".format(i, nodes_indexes[i]),
                    fontsize=13,
                    fontname="sans-serif")
        y -= 0.03
    y = 0.40
    for i in range(20, 30):
        plt.figtext(x=0.47,
                    y=y,
                    s="{} {}".format(i, nodes_indexes[i]),
                    fontsize=13,
                    fontname="sans-serif")
        y -= 0.03
    y = 0.40
    for i in range(30, 60):
        plt.figtext(x=0.64,
                    y=y,
                    s="{} {}".format(i, nodes_indexes[i]),
                    fontsize=13,
                    fontname="sans-serif")
        y -= 0.03
    fig.add_subplot(1, 1, 1)
    img = plt.imshow([cam], vmin=0, vmax=1, cmap="jet")
    plt.savefig(path_to_save)


def create_labels(state):
    switcher = {0 : "State1",
    1: "State2"}
    return switcher.get(state, lambda: "Invalid state")
