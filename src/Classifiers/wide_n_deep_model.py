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
import torch.functional as F
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
        pairs=[],
        remove_orig_nonserial=False
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
            pd.CategoricalDtype(categories=possible_categories)
        )
        converted = pd.get_dummies(cat, prefix=dom)
        df = pd.concat((df, converted))

    if remove_orig_nonserial:
        for dom in domain_dims.keys():
            del df[dom]
    return df


# ----------------------------------------------- #

class wide_n_deep(nn.Module):
    def __init__(
            self,
            wide_inp_01_dim=None,
            pretrained_node_embeddings=None,  # type : FloatTensor,
            tune_entity_emb=False,
            deep_FC_layer_dims=None,
            num_entities=None,  # total number of entities (reqd if no pretrained emb)
            entity_emb_dim=None,
            num_domains=None
    ):
        super(wide_n_deep, self).__init__()

        self.wide_inp_dim = wide_inp_01_dim
        self.wide_Linear = nn.Linear(wide_inp_01_dim, 1)

        if pretrained_node_embeddings is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(
                pretrained_node_embeddings,
                freeze=tune_entity_emb
            )
        else:
            self.embedding = torch.nn.Embedding(
                num_entities,
                entity_emb_dim
            )

        self.entity_emb_dim = entity_emb_dim
        self.num_domains = num_domains
        self.concat_emb_dim = self.num_domains * self.entity_emb_dim
        inp_dim = self.concat_emb_dim  # Concatenation
        self.deep_mlp = MLP(
            inp_dim,
            deep_FC_layer_dims
        )

        self.wide_inp_dim = wide_inp_01_dim
        # self.o_bias = torch.Variable(
        #     torch.FloatTensor([wide_op_dim + deep_FC_layer_dims[-1]])
        # )
        return

    def forward(self, input_x):

        x_wide = input_x[:, :self.wide_inp_dim]
        x_deep = input_x[:, self.wide_inp_dim:]
        x_deep = self.embedding(x_deep)
        x_deep = x_deep.view(-1, self.concat_emb_dim)
        x_d = self.deep_mlp(x_deep)
        x_o = self.wide_Linear(x_wide.float()) + x_d
        res = nn.functional.sigmoid(x_o)
        return res


def test():
    # ---
    # loss : BCELoss
    # ---
    model = wide_n_deep(
        wide_inp_01_dim=7,
        num_domains=3,
        entity_emb_dim=4,
        deep_FC_layer_dims=[5, 6],
        pretrained_node_embeddings=torch.FloatTensor(np.random.random([15, 4]))
    )

    x0 = np.random.randint(0, 2, size=[16, 7])
    x1 = np.random.randint(0, 2, size=[16, 3])
    x2 = np.hstack([x0, x1])
    x = LT(x2)

    print(' Input :: ', x.shape)
    y_pred = model(x)
    y_true = FT(np.random.randint(0, 2, size=[16, 1]))
    criterion = nn.BCELoss()
    loss = criterion(y_pred, y_true)
