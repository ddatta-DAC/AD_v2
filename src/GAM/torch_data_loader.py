#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler

import pandas as pd
import numpy as np
import torch

class  custom_dataset_type_1(Dataset):
    def __init__(
            self,
            file_path =None,
            is_numpy = False,
            is_csv = True
    ):
        if is_numpy:
            self.data = np.load(file_path)
        if is_csv:
            self.data = pd.read_csv(file_path, index_col=None)


    def __len__(self):
        return len(self.data )


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        res = self.data.iloc[idx]
        return np.array(res)



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
        sampler = RandomSampler(dataset)
    )

    arr = []
    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch)
        # print(sample_batched.shape)
        arr.extend(list(sample_batched[:,0]))
    print(len(arr))


    # rs = RandomSampler(
    #     dataset,
    #     False,
    #     100
    # )
    # for sample_batched in enumerate(rs):
    #     print(sample_batched)


test()