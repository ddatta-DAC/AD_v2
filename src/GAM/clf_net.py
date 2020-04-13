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
        for i in range(num_mlp_layers):
            if i == num_mlp_layers-1:
                op_dim = 2
            else:
                op_dim = layer_dimensions[i]
            self.mlp_layers[i] = nn.Linear(inp_dim, op_dim )
            self.register_parameter(
                'mlp_' + str(i),
                self.mlp_layers[i].weight
            )
            print(self.mlp_layers[i])
            inp_dim = op_dim

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.softmax_1 = nn.Softmax(dim=-1)
        return

    # Input is a tensor/array
    # shape [?, emb_size ]
    # Output :
    # shape : [ ?, output_dimension ]
    def forward(self, input_x):
        x = input_x
        for i in range(self.num_mlp_layers):
            x = self.dropout(x)
            x = self.mlp_layers[i](x)
            if i != self.num_mlp_layers-1 :
                x = self.activation(x)

        op_x = self.softmax_1(x)
        return op_x
