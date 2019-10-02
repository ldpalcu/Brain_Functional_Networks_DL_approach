from os import listdir
import os
from os.path import isfile, join
import numpy as np
import pprint
import matplotlib.pyplot as plt
import Arrays


def getdicts(directory):
    onlyfiles = [
        f for f in listdir(directory)
        if isfile(join(directory, f)) and f.endswith('.txt')
    ]

    filesdict = {}

    for f in onlyfiles:
        with open(directory + "/" + f, "r") as g:
            newarr = np.zeros(shape=85)
            for l in g:
                arr = l.rstrip("\n").split(" ")
                newarr[int(float(arr[0])) - 1] = float(arr[1])

            filesdict[f] = newarr

    return filesdict


state = []
order = range(0, 85)
labels = False

nr_dataset = 60
dataset_type = ""

dir_path_files = ""
dir_path_figures = ""

if __name__ == "__main__":

    for s in state:

        control = getdicts(dir_path_files + s)
        etoh = getdicts(dir_path_files + s)

        arr_dict = {}

        for k in control.keys():
            arr_dict[k] = control[k] - etoh[k]

        print(len(arr_dict.keys()))
        arr = np.zeros(shape=(len(arr_dict.keys()) - 2, 85))

        for k, v in arr_dict.items():
            _, _, _, _, nr_cam = k.split("_")
            nr_cam, _ = nr_cam.split(".t")
            if nr_cam != "high" and nr_cam != "low":
                nr_cam = int(float(nr_cam) * 10)
                arr[10 - nr_cam] = v

        orderd_ROIs = []
        for i in order:
            if labels:
                orderd_ROIs.append(Arrays.ROIs[i])
            else:
                orderd_ROIs.append(i)

        plt.figure(figsize=(20, 9))

        extent = [0, 85, 0.1, 1.1]

        im = plt.imshow(arr[0:10],
                        cmap="jet",
                        extent=extent,
                        interpolation="nearest",
                        aspect="auto",
                        vmin=-0.5,
                        vmax=0.5)

        # plt.colorbar(im, boundaries=np.arange(-2, 2.1, 0.1), aspect=30, spacing="uniform")
        plt.colorbar(im, aspect=30)
        plt.tight_layout()

        if labels:
            plt.gcf().subplots_adjust(bottom=0.25)

        if labels:
            plt.xticks(range(0, 85),
                       orderd_ROIs,
                       fontsize=8,
                       rotation=45,
                       ha="right")
        else:
            plt.xticks(range(0, 85), orderd_ROIs, fontsize=8)

        plt.yticks(np.arange(0.1, 1.1, 0.1))

        plt.xlabel("Node")
        plt.ylabel("Importance")

        plt.title("State1_State2_" + str(nr_dataset))

        # plt.show()
        plt.savefig(dir_path_figures + dataset_type + s + "_" +
                    str(nr_dataset) + ".png",
                    bbox_inches="tight")
        plt.close()