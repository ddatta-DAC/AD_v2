#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from src.Classifiers.MLP import MLP
import torch.nn.functional as F


def clf_loss(y_pred, y_true):
    if list(y_true.size())[1] == 1:
        y_true = y_true.squeeze(1)
    loss = F.nll_loss(y_pred, y_true)
    return loss


# -------------------------------------------- #
# MLP Binary Classifier
# -------------------------------------------- #
class clf_net_v2(nn.Module):
    def __init__(
            self,
            input_dim,
            mlp_layer_dims,
            dropout=0.05,
            activation='relu'
    ):
        super(clf_net_v2, self).__init__()

        self.mlp = MLP(
            input_dim,
            mlp_layer_dims,
            dropout=dropout,
            batch_norm_flag=False,
            output_layer=True,
            activation=activation,
            output_activation='sigmoid'
        )

        return

    def forward(self, input_x):
        x = input_x
        # Reshape the input
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-2] * x.shape[-1])

        op_x = self.mlp(x)

        return op_x
