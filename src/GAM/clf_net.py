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

    def setup_Net(
            self,
            inp_emb_dimension,
            layer_dimensions,
            dropout
    ):

        num_mlp_layers =  len(layer_dimensions)
        self.num_mlp_layers = num_mlp_layers
        self.mlp_layers = [None] * num_mlp_layers
        inp_dim = inp_emb_dimension
        for i in range(num_mlp_layers):
            self.mlp_layers[i] = nn.Linear(inp_dim, layer_dimensions[i])
            self.register_parameter('mlp_' + str(i), self.encoder[i].weight)
            print(self.mlp_layers[i])
            inp_dim = layer_dimensions[i]

        self.dropout = nn.Dropout(dropout)

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
                x = nn.LeakyReLU(x)

        op_x = nn.Sigmoid(x)
        return op_x

