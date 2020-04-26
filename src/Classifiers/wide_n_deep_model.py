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
from itertools import combinations

try:
    from .MLP import MLP
except:
    from MLP import MLP


def cross_feature_generator(df, f1, f2, dim1, dim2):
    pandarallel.initialize()
    h1 = max(23, 6 * int(math.sqrt(dim1 * dim2) // 6) - 1)
    h2 = max(7, 6 * int((h1 // 2) // 6) - 1, 7)
    if f1 > f2:  # Sorted lexicographically
        f1, f2 = f2, f1
        dim1, dim2 = dim2, dim1

    def feature_hash(x):
        return int((x + (x % h2)) % h1)

    def _cross_value(row):
        return str(int(row[f1])) + '_' + str(int(row[f2]))

    new_cross_col = f1 + '_' + f2
    df[new_cross_col] = df.parallel_apply(_cross_value, axis=1)

    # possible values
    value2hash_dict = {}
    possible_values = []
    for i in range(dim1):
        for j in range(dim2):
            possible_values.append(str(i) + '_' + str(j))

    for e in enumerate(possible_values, 0):
        value2hash_dict[e[1]] = feature_hash(e[0])

    # HashReplace value with hashed value
    def HashReplace(_row):
        row = _row.copy()
        row[new_cross_col] = value2hash_dict[row[new_cross_col]]
        return row

    df = df.parallel_apply(HashReplace, axis=1)
    possible_categories = list(range(h1))
    cat = pd.Series(list(df[new_cross_col]))
    cat = cat.astype(
        pd.CategoricalDtype(categories=possible_categories)
    )
    converted = pd.get_dummies(cat, prefix=new_cross_col)
    df = pd.concat((df, converted), axis=1)
    try:
        del df[new_cross_col]
    except:
        pass

    return df


def wide_N_deep_data_preprocess(
        df,
        domain_dims,
        pairs=[],
        remove_orig_nonserial=False,
        id_col = 'PanjivaRecordID'
):
    df = df.copy()
    if pairs is None or len(pairs) == 0:
        domains = list(domain_dims.keys())
        pairs = []
        for p in combinations(domains, 2):
            pairs.append(p)


    def parallel_comb(
        df,
        pair
    ):
        f1 = pair[0]
        f2 = pair[1]
        df_inp = pd.DataFrame(df[[id_col, f1, f2]], copy=True)
        df_op = cross_feature_generator(
            df_inp,
            f1, f2,
            domain_dims[f1],
            domain_dims[f2]
        )
        return (df_op, [f1,f2])



    from joblib import Parallel,delayed
    result = Parallel(n_jobs=10)(delayed(parallel_comb)(df,pair,) for pair in pairs)

    for item in result:
        _df = item[0]
        merge_cols = [id_col] + item[1]
        df =  df.merge(_df, on =merge_cols ,how = 'inner')

    # for pair in pairs:
    #     f1 = pair[0]
    #     f2 = pair[1]
    #     df_inp = pd.DataFrame(df[[id_col,f1,f2]],copy=True)
    #     df_op = cross_feature_generator(
    #         df_inp,
    #         f1, f2,
    #         domain_dims[f1],
    #         domain_dims[f2]
    #     )
    #     df = df.merge(df_op, on =[id_col,f1,f2],how='inner')

    # -----------------
    # Convert the regular domains to one-hot
    # -----------------
    for dom in domain_dims.keys():
        possible_categories = list(range(domain_dims[dom]))
        cat = pd.Series(list(df[dom]))
        cat = cat.astype(
            pd.CategoricalDtype(categories=possible_categories)
        )
        converted = pd.get_dummies(cat, prefix=dom)
        df = pd.concat((df, converted),axis=1)

    if remove_orig_nonserial:
        for dom in domain_dims.keys():
            del df[dom]
    return df


# ========================================= #
# Wide and Deep Module
# ========================================= #
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
        self.o_bias = nn.Parameter(
            torch.FloatTensor([1])
        )
        return

    def forward(self, input_x):

        x_wide = input_x[:, :self.wide_inp_dim]
        x_deep = input_x[:, self.wide_inp_dim:]
        x_deep = self.embedding(x_deep)
        x_deep = x_deep.view(-1, self.concat_emb_dim)
        x_d = self.deep_mlp(x_deep)
        x_o = self.wide_Linear(x_wide.float()) + x_d + self.o_bias
        res = torch.sigmoid(x_o)
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
