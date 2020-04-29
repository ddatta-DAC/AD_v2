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
from itertools import islice


# ================================
# This custom data set can load single labelled or unlabelled dataset.
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

        if self.y_col is not None:
            return x1, y1
        else:
            return x1


# ------------------------------------------------- #
# This generates a pair of (x1,[y1]), (x2,[y2])
# If allow_refresh = False : return None on exhausting one data set
# ------------------------------------------------- #

class pairDataGenerator_v1():
    def __init__(
            self,
            df_1,
            df_2,
            x_cols,
            batch_size=256,
            y1_col=None,
            y2_col=None,
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
        if not self.allow_refresh:
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
            return (x1, x2), (y1, y2)
        else:
            return (x1, x2)


class pairDataGenerator_v2():

    def __init__(
            self,
            df_1,
            df_2,
            x1_F_col,
            x2_F_col,
            x1_G_col,
            x2_G_col,
            y1_col=None,
            y2_col=None,
            batch_size=256,
            device=None,
            allow_refresh=False
    ):
        self.device = device
        self.x1_F_col = x1_F_col
        self.x2_F_col = x2_F_col
        self.x1_G_col = x1_G_col
        self.x2_G_col = x2_G_col
        self.y1_col = y1_col
        self.y2_col = y2_col
        self.batch_size = batch_size
        # Shuffle
        self.df_1 = df_1.reindex(np.random.permutation(df_1.index))
        self.df_2 = df_2.reindex(np.random.permutation(df_2.index))
        self.index_1 = np.random.permutation(len(self.df_1))
        self.index_2 = np.random.permutation(len(self.df_2))

        self.iter_obj1 = self.get_indices_iter(self.index_1, batch_size, allow_refresh)
        self.iter_obj2 = self.get_indices_iter(self.index_2, batch_size, allow_refresh)
        print(self.iter_obj1)
        print(self.iter_obj2)
        return

    def get_indices_iter(self, index_i, batch_size, allow_refresh):
        def next_indices(
                index_i,
                batch_size,
                refresh=False
        ):
            cur_idx = 0
            while cur_idx + batch_size < len(index_i):
                if refresh and cur_idx + batch_size >= len(index_i):
                    cur_idx = 0
                    np.random.shuffle(index_i)
                    print('---')
                yield index_i[cur_idx:cur_idx + batch_size]
                cur_idx += batch_size

        obj = iter(next_indices(index_i, batch_size, allow_refresh))
        return obj

    def get_next(self):
        try:
            next_1 = next(self.iter_obj1)
        except:
            next_1 = None

        try:
            next_2 = next(self.iter_obj2)
        except:
            next_2 = None

        return next_1, next_2


class pairDataGenerator_v2:

    def __init__(
            self,
            df_1,
            df_2,
            x1_F_col,
            x2_F_col,
            x1_G_col,
            x2_G_col,
            y1_col=None,
            y2_col=None,
            batch_size=256,
            device=None,
            allow_refresh=False
    ):
        self.device = device
        self.x1_F_col = x1_F_col
        self.x2_F_col = x2_F_col
        self.x1_G_col = x1_G_col
        self.x2_G_col = x2_G_col
        self.y1_col = y1_col
        self.y2_col = y2_col
        self.batch_size = batch_size
        # Shuffle
        df_1 = df_1.sample(frac=1)
        df_2 = df_2.sample(frac=1)
        self.df_1 = df_1.reset_index(drop=True)
        self.df_2 = df_2.reset_index(drop=True)

        index_1 = np.random.permutation(len(self.df_1))
        index_2 = np.random.permutation(len(self.df_2))

        self.iter_obj1 = self._get_indices_iter(index_1, batch_size, allow_refresh)
        self.iter_obj2 = self._get_indices_iter(index_2, batch_size, allow_refresh)

        return

    def _get_indices_iter(self, index_i, batch_size, allow_refresh):
        def next_indices(
                index_i,
                batch_size,
                refresh=False
        ):
            cur_idx = 0
            while True:
                if cur_idx + batch_size > len(index_i):
                    if allow_refresh:
                        cur_idx = 0
                        np.random.shuffle(index_i)
                    else:
                        break

                yield np.array(index_i[cur_idx:cur_idx + batch_size])
                cur_idx += batch_size

        obj = iter(next_indices(index_i, batch_size, allow_refresh))
        return obj

    def get_next(self):
        try:
            next_1_idx = next(self.iter_obj1)
        except StopIteration:
            next_1_idx = None

        try:
            next_2_idx = next(self.iter_obj2)
        except StopIteration:
            next_2_idx = None

        if next_1_idx is None or next_2_idx is None:
            return None

        x1_F = None
        x1_G = None
        x2_F = None
        x2_G = None
        y1 = None
        y2 = None

        if self.x1_F_col is not None:
            x1_F = LT(self.df_1.loc[next_1_idx, self.x1_F_col].values).to(self.device)

        if self.x1_G_col is not None:
            x1_G = LT(self.df_1.loc[next_1_idx, self.x1_G_col].values).to(self.device)

        if self.x2_F_col is not None:
            x2_F = LT(self.df_2.loc[next_2_idx, self.x2_F_col].values).to(self.device)

        if self.x2_G_col is not None:
            x2_G = LT(self.df_2.loc[next_2_idx, self.x2_G_col].values).to(self.device)

        if self.y1_col is not None:
            y1 = FT(self.df_1.loc[next_1_idx, self.y1_col].values).to(self.device)

        if self.y2_col is not None:
            y2 = FT(self.df_2.loc[next_2_idx, self.y2_col].values).to(self.device)

        return (x1_F, x1_G, y1), (x2_F, x2_G, y2)