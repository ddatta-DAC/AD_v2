import pandas as pd
import numpy as np
from torch import nn
import torch
from torch import tensor

import multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.data
import yaml
import pickle
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            input_dim,
            mlp_layer_dims,
            dropout = 0.05,
            batch_norm_flag  = False,
            output_layer=True,
            activation = 'relu'
    ):
        super().__init__()
        layers = list()
        for embed_dim in mlp_layer_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))

            if batch_norm_flag:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            if activation == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation == 'tanh':
                layers.append(torch.nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(torch.nn.Sigmoid())
            elif activation == 'selu':
                layers.append(torch.nn.SELU())
            elif activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU())

            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        return


    def forward(self, x):
        return self.mlp(x)
