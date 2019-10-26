from __future__ import print_function

import os
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import logging

sys.path.append('%s/lib' % os.path.dirname(os.path.realpath(__file__)))
from gnn_lib import GNNLIB
from pytorch_util import weights_init, gnn_spmm

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()
logger.disabled = True


class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30,
                 conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU'):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = int(sum(latent_dim))
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats + num_edge_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i - 1], latent_dim[i]))

        # self.conv1d_params1 = nn.Conv2d(1, conv1d_channels[0], 7, 2)
        # self.maxpool1d = nn.MaxPool2d(2, 2)
        # self.conv1d_params2 = nn.Conv2d(conv1d_channels[0], conv1d_channels[1], 5, 1)

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        # self.features = None
        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1] #2304

        self.gradients = None

        self.nodes_indexes = None

        if num_edge_feats > 0:
           self.w_e2l = nn.Linear(num_edge_feats, num_node_feats)

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        weights_init(self)

    def save_gradients(self, grad):
        self.gradients = grad

    def forward(self, graph_list, node_feat, edge_feat, output_features):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                edge_feat = edge_feat.cuda()
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs,
                                       output_features)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs,
                              output_features):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        # print("cur_message_layer before {}".format(cur_message_layer.size()))
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        logger.info("cur_message_layers final {}".format(cur_message_layer.size()))

        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(subg_sp.size()[0]):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k - k, self.total_latent_dim)
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]
            if output_features:
                self.nodes_indexes = topk_indices

        ''' traditional 1d convlution and dense layers '''
        logger.info(batch_sortpooling_graphs.size())
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        # to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k, self.total_latent_dim))
        logger.info("sort pooling layer {}".format(to_conv1d.size()))
        conv1d_res = self.conv1d_params1(to_conv1d)
        logger.info(self.conv1d_params1)
        conv1d_res = self.conv1d_activation(conv1d_res)
        logger.info(conv1d_res.size())
        # visualize activations
        conv1d_res = self.maxpool1d(conv1d_res)
        logger.info(conv1d_res.size())
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv_outputs = conv1d_res

        conv1d_res.register_hook(self.save_gradients)
        logger.info(conv1d_res.size())

        conv1d_res = self.conv1d_activation(conv1d_res)
        #conv_outputs = conv1d_res
        logger.info(conv1d_res.size())

        #conv1d_res.register_hook(self.save_gradients)

        # conv1d_res = self.maxpool1d(conv1d_res)
        # logger.info(conv1d_res.size())

        to_dense = conv1d_res.view(len(graph_sizes), -1)
        logger.info("to dense {}".format(to_dense.size()))

        # interpretability part
        final_layer_nodes = {}
        if output_features:
            k = 0
            for idx in range(0, int(len(self.nodes_indexes)/2)-5 + 1):
                final_layer_nodes.update({idx: self.nodes_indexes[k:(k+5*2)]})
                k += 2
            self.nodes_indexes = final_layer_nodes

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        if output_features:
            return conv_outputs, self.gradients, self.conv1d_activation(reluact_fp), self.nodes_indexes
        else:
            return self.conv1d_activation(reluact_fp)
