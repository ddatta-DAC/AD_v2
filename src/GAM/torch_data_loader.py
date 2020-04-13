#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import os
import sys
from torch import FloatTensor as FT
from torch import LongTensor as LT

sys.path.append('./../..')
sys.path.append('./..')
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler

import pandas as pd
import numpy as np
import torch


class custom_dataset_type_1(Dataset):
    def __init__(
            self,
            file_path=None,
            is_numpy=False,
            is_csv=True
    ):
        if is_numpy:
            self.data = np.load(file_path)
        if is_csv:
            self.data = pd.read_csv(file_path, index_col=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        res = self.data.iloc[idx]
        return np.array(res)


# ------------------------------------------------- #
# Given 2 data sets, retrieves samples from both in parallel
# must provide the x cols and y cols for both
# -------------------------------------------------- #
class pair_Dataset(Dataset):
    def __init__(
            self,
            df_1,
            df_2,
            x_cols,
            y_col=None,
            size_1=None,
            size_2=None,
            shuffle_prob = 0.25
    ):
        self.data_1 = df_1
        if size_1 is None:
            size_1 = len(self.data_1)
        else:
            self.data_1 = self.data_1.head(size_1)

        self.data_2 = df_2
        if size_2 is None:
            size_2 = len(self.data_2)
        else:
            self.data_2 = self.data_2.head(size_2)
        self.size_1 = size_1
        self.size_2 = size_2
        self.x_cols = x_cols
        self.y_col = y_col
        self.shuffle_prob = shuffle_prob
        return

    def __len__(self):
        return max(self.size_1, self.size_2)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        min_size = min(self.size_1, self.size_2)
        try:
            idx_s = [_ for _ in map(lambda x: x % min_size, idx)]
        except:
            idx_s = idx % min_size
        shuffle_flag = np.random.uniform(0,1) <= self.shuffle_prob

        if self.size_1 < self.size_2:
            if shuffle_flag:
                np.random.shuffle(self.data_1.values)
            idx_1 = idx_s
            idx_2 = idx
        elif self.size_1 > self.size_2:
            if shuffle_flag:
                np.random.shuffle(self.data_2.values)
            idx_1 = idx
            idx_2 = idx_s
        else:
            idx_1 = idx
            idx_2 = idx

        res_1_x = self.data_1[self.x_cols].iloc[idx_1]
        res_2_x = self.data_2[self.x_cols].iloc[idx_2]

        x1 = np.array(res_1_x)
        x2 = np.array(res_2_x)
        if self.y_col is not None:
            try:
                y1 = np.array(self.data_1[self.y_col].iloc[idx_1])
            except:
                y1 = None
            try:
                y2 = np.array(self.data_2[self.y_col].iloc[idx_2])
            except:
                y2 = None

            return (x1, x2), (y1, y2)
        else:
            return x1, x2


# ================================
#
# ================================
class type1_Dataset(Dataset):
    def __init__(
            self,
            df,
            x_cols,
            y_col=None
    ):
        self.df = df
        self.x_cols = x_cols
        self.y_col = y_col


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.df[self.x_cols].iloc[idx]
        x = np.array(x)
        if self.y_col is not None:
            y = self.df[self.y_col].iloc[idx]
            y = np.array(y)
            y = np.reshape(y,[-1])

            return (x,y)
        return x

# ------------------------------------------------- #

def test():
    f_path = './../../generated_data_v1/us_import1/test_data.csv'
    dataset = custom_dataset_type_1(
        f_path
    )
    import multiprocessing
    num_proc = multiprocessing.cpu_count()
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=num_proc,
        sampler=RandomSampler(dataset)
    )

    arr = []
    for i_batch, sample_batched in enumerate(dataloader):
        arr.extend(list(sample_batched[:, 0]))
    print(len(arr))

    # rs = RandomSampler(
    #     dataset,
    #     False,
    #     100
    # )
    # for sample_batched in enumerate(rs):
    #     print(sample_batched)

    # dataloader1 = DataLoader(
    #     ds,
    #     batch_size=16,
    #     shuffle=False,
    #     num_workers=num_proc,
    #     sampler=RandomSampler(ds)
    # )


# test()
