#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

import torch
from torch import nn
from pandarallel import pandarallel

pandarallel.initialize()
import numpy as np
import os
import pandas as pd
import sys
import pickle
import argparse
import math
import  torch.functional as F
from torch import FloatTensor as FT
from torch import LongTensor as LT

try:
    from .MLP import MLP
except:
    from MLP import MLP


def cross_feature(df, f1, f2, dim1, dim2):
    h1 = 6 * int(math.sqrt(dim1 * dim2) // 6) - 1
    h2 = 6 * int((h1 // 2) // 6) + 1
    print(h1, h2)

    def feature_hash(x):
        return int((x + x % h2) % h1)

    def _cross(row):
        return str(row[f1]) + '_' + str(row[f2])

    new_col = f1 + '_' + f2
    df[new_col] = df.parallel_apply(_cross)
    _dict = {}
    for k, v in enumerate(set(df[new_col])):
        _dict[v] = feature_hash(k)

    # replace
    def replace(_row):
        row = _row.copy()
        row[new_col] = _dict[row[new_col]]
        return row

    df = df.parallel_apply(replace, axis=1)
    df = pd.get_dummies(
        df, columns=[new_col]
    )
    return df


def preprocess(
        df,
        domain_dims,
        pairs = [],
        remove_orig_nonserial = False
):
    df = df.copy()
    for pair in pairs:
        f1 = pair[0]
        f2 = pair[1]
        df = cross_feature(df, f1, f2, domain_dims[f1], domain_dims[f2])
    # convert the regular domains to one-hot

    for dom in domain_dims.keys():
        possible_categories = list(range(domain_dims[dom]))
        cat = pd.Series(list(df[dom]))
        cat = cat.astype(
            pd.CategoricalDtype(categories = possible_categories)
        )
        converted = pd.get_dummies(cat,prefix = dom)
        df = pd.concat((df,converted))

    if remove_orig_nonserial:
        for dom in domain_dims.keys():
            del df[dom]
    return df



# ----------------------------------------------- #

class wide_n_deep(nn.Module):
    def __init__(
            self,
            deep_FC_layer_inp_dim,
            deep_FC_layer_dims,
            wide_inp_dim,
            wide_op_dim
    ):
        super(wide_n_deep).__init__()
        self.num_deep_FC_layers = len(deep_FC_layer_dims)

        self.deep_mlp = MLP(
            deep_FC_layer_inp_dim,
            deep_FC_layer_dims,
            batch_norm_flag = True
        )

        self.wide_inp_dim = wide_inp_dim
        self.wide_Linear = nn.Linear(wide_inp_dim,wide_op_dim)
        self.o_bias = torch.Variable(
            torch.FloatTensor([ wide_op_dim + deep_FC_layer_dims[-1]])
        )
        return


    def forward(self, input_x):
        w_indices = list(range(self.wide_inp_dim))
        x_wide = input_x[:,:w_indices]
        x_deep = input_x[:,w_indices:]

        x_w = self.wide_Linear(x_wide)
        x_d = x_deep
        for i in range(self.num_deep_FC_layers):
            x_d = self.deep_FC_layers[i](x_d)
            x_d = nn.functional.relu(x_d)

        x_o = x_w + x_d + self.o_bias

        res = nn.functional.sigmoid(x_o)
        return res

# ---
# loss : BCELoss
# ---

