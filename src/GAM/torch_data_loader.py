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
            batch_size=128,
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




class balanced_pair_Generator_v2:

    def __init__(
            self,
            df,
            x_col=None,
            y_col=None,
            batch_size=256,
            device=None,
            allow_refresh=False,
            id_col = 'PanjivaRecordID'
    ):
        self.device = device

        self.x_col = x_col
        self.y_col = y_col
        self.df = df
        self.batch_size = batch_size
        # Shuffle

        concordant_pairs = [] # 00, 11
        discordant_pairs = [] # 01

        ones = list(df.loc[df[y_col] ==1 ].index)
        print(len(ones))

        zeros =  list(df.loc[df[y_col] == 0 ].index)
        from itertools import combinations
        one_one = []
        for pr in combinations(ones,2):
            one_one.append(pr)

        # one_one = one_one [ np.random.choice(
        #     list(range(len(one_one))),
        #     size = max(25000,  len(one_one)*10//100),
        #     replace=False
        # )]
        one_one = np.array(one_one)
        _samples =  min(len(df)*10,  (one_one.shape[0]*20)//100)
        idx = list(np.random.randint(len(one_one), size=_samples))

        one_one = one_one[idx,:]
        zero_zero = []
        np.random.shuffle(zeros)
        count = 0
        for pr in combinations(zeros,2):
            zero_zero.append(pr)
            count += 1
            if count > one_one.shape[0] : break


        np.random.shuffle(zero_zero)
        zero_zero = np.array(zero_zero, dtype=int)
        zero_zero = zero_zero[:one_one.shape[0],:]

        concordant_pairs = np.vstack([one_one, zero_zero])
        max_len = concordant_pairs.shape[0]

        one_i = np.random.choice(ones, max_len, replace=True)
        np.random.shuffle(one_i)
        zeros_j = np.random.choice(zeros, max_len, replace=True)
        np.random.shuffle(zeros_j)

        for i,j in zip(one_i, zeros_j):
             discordant_pairs.append([i,j])

        all_pairs = np.vstack([concordant_pairs, discordant_pairs])
        all_pairs = np.array(all_pairs, dtype=int)
        np.random.shuffle(all_pairs)
        self.batch_count = all_pairs.shape[0]//batch_size
        self.iter_obj1 = self._get_indices_iter(all_pairs, batch_size, allow_refresh)
        return

    def get_num_batches(self):
        return  self.batch_count


    def _get_indices_iter(self, index_i, batch_size, allow_refresh=False):
        def next_indices(
                index_i,
                batch_size,
                allow_refresh=allow_refresh
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
        next_idx_1 = None
        next_idx_2 = None
        try:
            next_idx = next(self.iter_obj1)
        except StopIteration:
            next_idx = None

        next_idx_1 = next_idx[:,0]
        next_idx_2 = next_idx[:,1]


        if next_idx_1 is None or next_idx_2 is None:
            return None

        x1 = None
        x2 = None
        y1 = None
        y2 = None

        x1 = LT(self.df.loc[next_idx_1, self.x_col].values).to(self.device)
        y1 = LT(self.df.loc[next_idx_1, self.y_col].values).to(self.device)

        x2 = LT(self.df.loc[next_idx_2, self.x_col].values).to(self.device)
        y2 = LT(self.df.loc[next_idx_2, self.y_col].values).to(self.device)

        return (x1, y1), (x2, y2)

