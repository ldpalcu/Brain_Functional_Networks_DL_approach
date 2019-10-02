import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
from scipy.ndimage import interpolation


class GradCam:
    def __init__(self, model):
        self.model = model
        self.nodes_indexes = None
        self.labels = None
        self.pred_labels = None
        self.confidence_score = None
        self.model.eval()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        conv_output, gradients, model_output, nodes_indexes, labels = self.model.output_features(input)

        self.nodes_indexes = {k: nodes_indexes[k].data.numpy() for k, v in nodes_indexes.items()}
        self.labels = labels.data.numpy()[0]

        if index is None:
            index = np.argmax(model_output.data.numpy())

        self.pred_labels = index

        self.confidence_score = model_output.data.numpy()[0]

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][index] = 1
        # # Zero grads
        self.model.mlp.zero_grad()
        # #Backwards pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # #Get hooked gradients
        guided_gradients = gradients.data.numpy()[0]
        # #Get conv outputs
        target = conv_output.data.numpy()[0, :]

        # Get weights from gradients
        # Take average for each gradient
        weights = np.mean(guided_gradients, axis=1)
        #print("weights {}".format(weights.shape))
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # print("cam {}".format(cam.shape))

        for i, w in enumerate(weights):
            cam += w * target[i, :]

        cam_i = np.maximum(cam, 0)

        cam_i = cam_i - np.min(cam_i) 
        cam_i = cam_i / np.max(cam_i)

        return cam_i
