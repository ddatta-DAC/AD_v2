#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import os
import copy
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append('./../..')
sys.path.append('./..')
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torch import FloatTensor as FT
from torch import LongTensor as LT

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
            shuffle_prob=0.25
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
        shuffle_flag = np.random.uniform(0, 1) <= self.shuffle_prob

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
# This custom dataset can load single labelled or unlabelled dataset.
# ================================
class type1_Dataset(Dataset):
    def __init__(
            self,
            df,
            x_cols,
            y_col=None,
            return_id_col=False
    ):
        self.id_col = 'PanjivaRecordID'
        self.df = df
        self.x_cols = x_cols
        self.y_col = y_col
        self.return_id_col = return_id_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.df[self.x_cols].iloc[idx]
        x = np.array(x)
        id = None
        if self.return_id_col:
            id = np.array(self.df[self.id_col].iloc[idx])

        if self.y_col is not None:
            y = self.df[self.y_col].iloc[idx]
            y = np.array(y)
            y = np.reshape(y, [-1])
            if id is not None:
                return (id, x, y)
            else:
                return (x, y)
        elif id is not None:
            return (id, x)
        else:
            return x


# ------------------------------------------------- #

# ================================
# This generates a single of x1,[y1]
# ================================
class singleDataGenerator():
    def __init__(self,
                 df,
                 x_cols,
                 batch_size=256,
                 y_col=None,
                 num_workers=0
        ):
        self.x_cols = x_cols
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.y_col = y_col

        self.ds1 = type1_Dataset(
            df,
            x_cols,
            y_col
        )
        self.iter_obj1 = iter(self.get_dataloader(self.ds1))

        return

    def get_dataloader(
            self,
            ds
    ):
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            sampler=RandomSampler(ds),
            drop_last=True
        )
        return dl

    def get_next(self):
        y1 = None

        try:
            x1_y1 = next(self.iter_obj1)
        except StopIteration:
            return None

        if self.y_col is not None:
            x1 = x1_y1[0]
            y1 = x1_y1[1]
        else:
            x1 = x1_y1

        if self.y_col is not None :
            return x1, y1
        else:
            return  x1



data_LL_generator = pairDataGenerator(
                df_1 = df_L,
                df_2 = df_L,
                x_cols=g_feature_cols,
                y1_col= None,
                y2_col= label_col,
                batch_size=batch_size_r)

# ------------------------------------------------- #
# This generates a pair of (x1,[y1]), (x2,[y2])
# If allow_refresh = False : return None on exhausting one data set
# ------------------------------------------------- #

class pairDataGenerator():
    def __init__(
            self,
            df_1,
            df_2,
            x_cols,
            batch_size=256,
            y1_col =None,
            y2_col =None,
            num_workers=0
    ):
        self.x_cols = x_cols
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ds1 = type1_Dataset(
            df_1,
            x_cols,
            y1_col
        )

        self.ds2 = type1_Dataset(
            df_2,
            x_cols,
            y2_col
        )

        self.iter_obj1 = iter(self.get_dataloader(self.ds1))
        self.iter_obj2 = iter(self.get_dataloader(self.ds2))
        self.y1_col = y1_col
        self.y2_col = y2_col
        self.allow_refresh = False

        return
    def set_allow_refresh(self, flag=True):
        self.allow_refresh = flag

    def get_dataloader(
            self,
            ds
    ):
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            sampler=RandomSampler(ds),
            drop_last=True
        )
        return dl

    def noRefresh_get_next(self):
        y1 = None
        y2 = None
        try:
            x1_y1 = next(self.iter_obj1)
        except StopIteration:
            return None
        try:
            x2_y2 = next(self.iter_obj2)
        except StopIteration:
            return None
        if self.y1_col is not None:
            x1 = x1_y1[0]
            y1 = x1_y1[1]
        else:
            x1 = x1_y1
        if self.y2_col is not None:
            x2 = x2_y2[0]
            y2 = x2_y2[1]
        else:
            x2 = x2_y2
        if self.y1_col is not None or self.y2_col is not None:
            return (x1, x2), (y1, y2)
        else:
            return (x1, x2)


    def get_next(self):
        y1 = None
        y2 = None
        if not self.allow_refresh :
            return self.noRefresh_get_next()

        try:
            if self.y1_col is not None:
                x1_y1 = next(self.iter_obj1)
                x1 = x1_y1[0]
                y1 = x1_y1[1]
            else:
                x1 = next(self.iter_obj1)
        except StopIteration:
            self.iter_obj1 = iter(self.get_dataloader(self.ds1))

            if self.y1_col is not None:
                x1_y1 = next(self.iter_obj1)
                x1 = x1_y1[0]
                y1 = x1_y1[1]
            else:
                x1 = next(self.iter_obj1)

        try:
            if self.y2_col is not None:
                x2_y2 = next(self.iter_obj2)
                x2 = x2_y2[0]
                y2 = x2_y2[1]
            else:
                x2 = next(self.iter_obj2)
        except StopIteration:
            self.iter_obj2 = iter(self.get_dataloader(self.ds2))
            if self.y2_col is not None:
                x2_y2 = next(self.iter_obj2)
                x2 = x2_y2[0]
                y2 = x2_y2[1]
            else:
                x2 = next(self.iter_obj2)

        if self.y1_col is not None or self.y2_col is not None:
            return (x1,x2), (y1,y2)
        else:
            return (x1, x2)



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
