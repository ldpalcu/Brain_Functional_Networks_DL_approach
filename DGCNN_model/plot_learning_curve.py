from matplotlib import pyplot as plt
import numpy as np

path_to_read = "./"
path_to_save = "./"

all_scores_train = []
all_scores_test = []
max_value = 75
int_value = 75
x_values = list(range(0, max_value))
for i in range(1, 11):
    file_train = "all_avg_loss_" + str(i) + ".txt"
    file_test = "all_test_loss_" + str(i) + ".txt"

    scores_train = (np.loadtxt(path_to_read + file_train)).tolist()[0: int_value]
    scores_test = (np.loadtxt(path_to_read + file_test)).tolist()[0: int_value]
    all_scores_train.append(scores_train)
    all_scores_test.append(scores_test)

    plt.xlabel("Num epochs")
    plt.ylabel("Score")
    plt.plot(x_values, scores_train, label="train")
    plt.plot(x_values, scores_test, label="test")
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig(path_to_save + "learning_curve_fold_" + str(i) + ".png")
    plt.clf()

mean_scores_train = sum(np.asarray(all_scores_train)) / 10
mean_scores_test = sum(np.asarray(all_scores_test)) / 10

plt.xlabel("Num epochs")
plt.ylabel("Score")
plt.plot(x_values, mean_scores_train, label="train")
plt.plot(x_values, mean_scores_test, label="test")
plt.ylim([0, 1])
plt.legend()
plt.savefig(path_to_save + "learning_curve_fold_" + "mean" + ".png")
plt.clf()
