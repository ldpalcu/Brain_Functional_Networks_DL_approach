import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import math
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier, MLPRegression
from sklearn import metrics
from util import cmd_args, load_data
from grad_cam import GradCam
# from visualize_cam import create_heatmap_cam_1d
# from visualize_cam import create_heatmap_cam_2d
from visualize_cam import create_labels


class Classifier(nn.Module):
    def __init__(self, regression=False):
        super(Classifier, self).__init__()
        self.regression = regression
        if cmd_args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        if cmd_args.gm == 'DGCNN':
            self.gnn = model(latent_dim=cmd_args.latent_dim,
                             output_dim=cmd_args.out_dim,
                             num_node_feats=cmd_args.feat_dim + cmd_args.attr_dim,
                             num_edge_feats=cmd_args.edge_feat_dim,
                             k=cmd_args.sortpooling_k,
                             conv1d_activation=cmd_args.conv1d_activation)
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            if cmd_args.gm == 'DGCNN':
                out_dim = self.gnn.dense_dim
            else:
                out_dim = cmd_args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class,
                                 with_dropout=cmd_args.dropout)
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=cmd_args.hidden, with_dropout=cmd_args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_features is not None:
            node_feat_flag = True
            concat_feat = []
        else:
            node_feat_flag = False

        if cmd_args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_feat_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
                concat_feat.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_feat_flag == True:
            node_feat = torch.cat(concat_feat, 0)

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags (node labels) with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features

        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels
        return node_feat, labels

    def forward(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat, False)
        mlp_res = self.mlp(embed, labels)
        return mlp_res

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        conv_outputs, gradients, embed, nodes_indexes = self.gnn(batch_graph, node_feat, edge_feat, True)
        logits, _, _, _ = self.mlp(embed, labels)
        # pred = logits.data.max(1, keepdim=True)[1]

        return conv_outputs, gradients, logits, nodes_indexes, labels


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize: (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss = classifier(batch_graph)
            all_scores.append(pred.cpu().detach())  # for binary classification
        else:
            logits, loss, acc, conf_mx = classifier(batch_graph)
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae))
            total_loss.append(np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc))
            total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()

    np.savetxt('test_scores.txt', all_scores)  # output test predictions

    if not classifier.regression:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))

    return avg_loss, conf_mx


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    train_graphs, test_graphs = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    if cmd_args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
        cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
        cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
        print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    all_avg_loss = []
    all_test_loss = []
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss, conf_mx_train = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, avg_loss[0], avg_loss[1], avg_loss[2]))
        all_avg_loss.append(avg_loss[1])

        classifier.eval()
        test_loss, conf_mx_test = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, test_loss[0], test_loss[1], test_loss[2]))
        all_test_loss.append(test_loss[1])

    with open('acc_results.txt', 'a+') as f:
        f.write(str(test_loss[1]) + '\n')

    if cmd_args.printAUC:
        with open('auc_results.txt', 'a+') as f:
            f.write(str(test_loss[2]) + '\n')

    np.savetxt("all_avg_loss_" + str(cmd_args.fold) + ".txt", all_avg_loss, "%.4f")
    np.savetxt("all_test_loss_" + str(cmd_args.fold) + ".txt", all_test_loss, "%.4f")

    # print(conf_mx_train)
    np.savetxt("conf_mx_train_" + str(cmd_args.fold) + ".txt", conf_mx_train, "%d")
    # print(conf_mx_test)
    np.savetxt("conf_mx_test_" + str(cmd_args.fold) + ".txt", conf_mx_test, "%d")

    # ################################# GRAD-CAM ################################################

    dir_path_figures = ""

    dir_path_files = ""

    grad_cam = GradCam(model=classifier)

    k = 0
    for test_graph in test_graphs:
        cam = grad_cam([test_graph], None)
        label = create_labels(grad_cam.labels)
        pred_label = create_labels(grad_cam.pred_labels)

        if grad_cam.labels == grad_cam.pred_labels:
            file_name_figure = "cam_" + str(k) + ".png"
            path_to_save_figures = os.path.join(dir_path_figures, label, "Good", file_name_figure)
            file_name_text = "cam_" + str(k) + ".txt"
            path_to_save_text = os.path.join(dir_path_files, label, "Good", file_name_text)
            file_name_pred = "cam_pred_" + str(k) + ".txt"
            path_to_save_pred = os.path.join(dir_path_files, label, "Good", file_name_pred)
            file_name_nodes = "nodes_" + str(k) + ".txt"
            path_to_save_nodes = os.path.join(dir_path_files, label, "Good", file_name_nodes)
        else:
            file_name_figure = pred_label + "_cam_" + str(k) + ".png"
            path_to_save_figures = os.path.join(dir_path_figures, label, "Wrong", file_name_figure)
            file_name_text = pred_label + "_cam_" + str(k) + ".txt"
            path_to_save_text = os.path.join(dir_path_files, label, "Wrong", file_name_text)
            file_name_pred = pred_label + "_cam_pred_" + str(k) + ".txt"
            path_to_save_pred = os.path.join(dir_path_files, label, "Wrong", file_name_pred)
            file_name_nodes = pred_label + "_nodes_" + str(k) + ".txt"
            path_to_save_nodes = os.path.join(dir_path_files, label, "Wrong", file_name_nodes)

        if not np.isnan(cam[0]):
            np.savetxt(path_to_save_text, cam, "%.4f")
            np.savetxt(path_to_save_pred, grad_cam.confidence_score, "%.4f")
            dict_to_write = np.empty((38, 10))
            for key, value in grad_cam.nodes_indexes.items():
                dict_to_write[key, :] = value
            np.savetxt(path_to_save_nodes, dict_to_write, "%d")
        create_heatmap_cam_1d_new(cam, path_to_save_figures, grad_cam.nodes_indexes, grad_cam.confidence_score)
        k += 1
