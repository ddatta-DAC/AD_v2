#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

import torch
import torch.nn as nn
from random import shuffle
import numpy as np
import sys
from torch.nn import Parameter
from torch import tensor

sys.path.append('./..')
sys.path.append('./../..')
import torch.nn.functional as F

# ----------------------------------------
# clf_net_v1 :: a simple MLP classifier
# ----------------------------------------
def clf_loss_v1 (y_pred, y_true):
    loss_func = nn.NLLLoss(reduce=None)
    y_true = y_true.squeeze(1)
    loss = loss_func(y_pred, y_true )
    return loss.mean()

class clf_net_v1(nn.Module):
    def __init__(
            self,
            inp_emb_dimension,
            layer_dimensions,
            dropout=0.05
    ):
        super(clf_net_v1, self).__init__()
        self.inp_emb_dimension = inp_emb_dimension
        self.setup_Net(
            inp_emb_dimension,
            layer_dimensions,
            dropout
        )
        return

    # ---------------------
    # Structure of MLP
    # [ inp_emb_dimension, [layer_dimensions] , 2 ]
    def setup_Net(
            self,
            inp_emb_dimension,
            layer_dimensions,
            dropout
    ):
        print(' Classifier module ')
        num_mlp_layers =  len(layer_dimensions) + 1
        self.num_mlp_layers = num_mlp_layers
        self.mlp_layers = [None] * num_mlp_layers
        inp_dim = inp_emb_dimension

        self.mlp_layers_1 = nn.Linear(inp_dim, layer_dimensions[0])
        self.mlp_layers_2 = nn.Linear(layer_dimensions[0], layer_dimensions[1])
        self.mlp_layers_3 = nn.Linear(layer_dimensions[1], layer_dimensions[2])
        self.mlp_layers_4 = nn.Linear(layer_dimensions[2], 2)
        self.dropout = nn.Dropout(dropout)
        self.activation_1 = torch.nn.Tanh()
        self.activation_2 = torch.nn.LeakyReLU()
        self.softmax_1 = nn.Softmax(dim=-1)
        return

    # Input is a tensor/array
    # shape [?, emb_size ]
    # Output :
    # shape : [ ?, output_dimension ]
    def forward(self, input_x):

        x = input_x
        x = self.mlp_layers_1(x)
        x = self.dropout(x)
        x = self.mlp_layers_2(x)

        x = self.dropout(x)
        x = self.mlp_layers_3(x)
        x = self.activation_1(x)
        x = self.mlp_layers_4(x)
        x = self.activation_1(x)

        op_x = self.softmax_1(x)
        return op_x



# -------------------------------------------- #
# MLP Classifier
# Configurable number of layers
# -------------------------------------------- #
class clf_net_v2(nn.Module):
    def __init__(
            self,
            inp_emb_dimension,
            layer_dimensions,
            dropout=0.05
    ):
        super(clf_net_v2, self).__init__()
        self.inp_emb_dimension = inp_emb_dimension
        self.setup_Net(
            inp_emb_dimension,
            layer_dimensions,
            dropout
        )
        return

    # ---------------------
    # Structure of MLP
    # [ inp_emb_dimension, [layer_dimensions] , 2 ]
    def setup_Net(
            self,
            inp_emb_dimension,
            layer_dimensions,
            dropout
    ):
        print(' Classifier module ')
        num_mlp_layers =  len(layer_dimensions) + 1
        self.num_mlp_layers = num_mlp_layers

        self.mlp_layers = nn.ModuleList()
        inp_dim = inp_emb_dimension
        for i in range(num_mlp_layers):
            if i == num_mlp_layers-1:
                op_dim = 2
            else:
                op_dim = layer_dimensions[i]
            self.mlp_layers.append(nn.Linear(inp_dim, op_dim))
            inp_dim = op_dim
        print('Number of MLP layers : {}'.format(self.mlp_layers), self.num_mlp_layers)

        self.dropout = nn.Dropout(dropout)
        self.activation_1 = torch.nn.Tanh()
        self.activation_2 = torch.nn.LeakyReLU()
        self.softmax_1 = nn.Softmax(dim=-1)

        return

    # Input is a tensor/array
    # shape [?, emb_size ]
    # Output :
    # shape : [ ?, output_dimension ]
    def forward(self, input_x):

        x = input_x
        # Reshape the input
        if len(x.shape)>2 :
            x = x.view(-1, x.shape[-2] * x.shape[-1])

        for i in range(self.num_mlp_layers):
            if i > 0 :
                x = self.dropout(x)

            x = self.mlp_layers[i](x)
            x = self.activation_1(x)

        op_x = self.softmax_1(x)
        return op_x

