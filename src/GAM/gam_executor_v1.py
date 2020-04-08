#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

import pandas as pd
import numpy as np
import os
import sys
from pandarallel import pandarallel
pandarallel.initialize()
sys.path.append('./../..')
sys.path.append('./..')
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
import yaml
import pickle
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import tensor
from .gam_module import gam_net
from .gam_module import gam_loss
from .clf_net import clf_net_v1 as clf_net
from .clf_net import clf_loss
from .record_node import graph_net_v1 as graph_net
from .torch_data_loader import pair_Dataset
from .torch_data_loader import type1_Dataset
try:
    print('Cuda available ::', torch.cuda.is_available(), 'Cde current device ::', torch.cuda.current_device(),
          torch.cuda.get_device_name(0))
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
except:
    print('No CUDA')


config_file = 'config.yaml'
CONFIG = None
DATA_SOURCE_DIR_1 = None
DATA_SOURCE_DIR_2 = None
model_use_data_DIR = None
DIR = None
domain_dims = None
score_col = 'score'
fraud_col = 'fraud'
anomaly_col = 'anomaly'
id_col = 'PanjivaRecordID'
label_col = 'y'
true_label_col = 'y_true'
feature_col_list = []
serial_mapping_df = None
is_labelled_col = 'labelled'

# =================================================

def setup_config(_DIR):
    global CONFIG
    global config_file
    global DATA_SOURCE_DIR_1
    global DATA_SOURCE_DIR_2
    global DIR
    global model_use_data_DIR
    global domain_dims
    global feature_col_list
    global serial_mapping_df
    global serialized_feature_col_list

    if _DIR is not None:
        DIR = _DIR

    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)

    DATA_SOURCE_DIR_1 = CONFIG['DATA_SOURCE_DIR_1']
    DATA_SOURCE_DIR_2 = CONFIG['DATA_SOURCE_DIR_2']

    DATA_SOURCE_DIR_1 = os.path.join(DATA_SOURCE_DIR_1, DIR)
    DATA_SOURCE_DIR_2 = os.path.join(DATA_SOURCE_DIR_1, DIR)

    model_use_data_DIR = CONFIG['model_use_data_DIR']
    if not os.path.exists(model_use_data_DIR): os.mkdir(model_use_data_DIR)
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    if not os.path.exists(model_use_data_DIR): os.mkdir(model_use_data_DIR)

    with open(
            os.path.join(
                DATA_SOURCE_DIR_1,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)

    feature_col_list = list(sorted(domain_dims.keys()))
    serialized_feature_col_list = [ '_' + _ for _ in feature_col_list]
    serial_mapping_df_path = os.path.join(
        CONFIG['serial_mapping_df_loc'],
        DIR,
        CONFIG['serial_mapping_df_name']
    )
    serial_mapping_df = pd.read_csv(serial_mapping_df_path, index_col=None)
    return


def set_ground_truth_labels(df):
    global true_label_col
    global fraud_col

    def aux_true_label(row):
        if row[fraud_col]:
            return 1
        else:
            return -1

    df[true_label_col] = df.parallel_apply(aux_true_label, axis=1)
    return df

# -----
# Return part of dataframe , where instances are labelled
# -----
def extract_labelled_df(df):
    global is_labelled_col
    res = pd.DataFrame(
        df.loc[df[is_labelled_col]==True],
        copy = True
    )
    return res

def extract_unlabelled_df(df):
    global is_labelled_col
    res = pd.DataFrame(
        df.loc[df[is_labelled_col]==False],
        copy = True
    )
    return res


def read_scored_data():
    global score_col
    global DATA_SOURCE_DIR_2
    global label_col

    df = pd.read_csv(
        os.path.join(DATA_SOURCE_DIR_2, 'scored_test_data.csv'), index_col=None
    )
    df = df.sort_values(by=[score_col])
    df[label_col] = 0
    df = set_ground_truth_labels(df)
    return df


def set_label_in_top_perc( df, perc):
    global score_col
    global true_label_col
    global id_col

    df = df.sort_values(by=[score_col])
    if perc > 1 :
        perc = perc/100
    count = len(df) * perc

    cand = list(df.head(count)[id_col])
    df.loc[df[id_col].isin(cand), label_col] =  df.loc[df[id_col].isin(cand), true_label_col]
    df.loc[df[id_col].isin(cand), is_labelled_col] = True
    return df

def convert_to_serial_IDs(
        df ,
        keep_entity_ids = False
):
    global feature_col_list
    global serial_mapping_df

    # Inplace conversion
    def aux_conv_toSerialID(_row):
        row = _row.copy()
        for fc in feature_col_list:
            col_name = fc
            if keep_entity_ids:
                col_name = '_' + fc
            row[col_name] = list(serial_mapping_df.loc[
                (serial_mapping_df['Domain']==fc) &
                (serial_mapping_df['Entity_ID'] == row[fc])
            ])[0]
        return row

    df = df.parallel_apply(aux_conv_toSerialID, axis=1)
    return df


# ------------------------------------------------- #

'''
1. Train g
2. Train f
3. Add in the most confident labels
'''


class net(nn.Module):
    def __init(self):
        super(net, self).__init__()
        # valid values for train_mode are 'f', 'g', False
        self.train_mode = False
        self.test_mode = False

    def setup_Net(
            self,
            node_emb_dimension,
            num_domains,
            output_dimensions,
            matrix_pretrained_node_embeddings,
            record_input_dimension,
            gam_encoder_dimensions,
            clf_inp_emb_dimension,
            clf_layer_dimensions
    ):
        self.graph_net = graph_net(
            node_emb_dimension,
            num_domains,
            output_dimensions,
            matrix_pretrained_node_embeddings
        )
        self.gam_net = gam_net(
            record_input_dimension,
            gam_encoder_dimensions,
        )
        self.clf_net = clf_net(
            clf_inp_emb_dimension,
            clf_layer_dimensions
        )



    # ---------------------------
    # Input should be [ Batch, record( list of entities ) ]
    # record( list of entities ) should have serialized entity id
    # ---------------------------
    # input_xy is an list
    def forward (
            self, input_x
    ):

        # ----------------
        # Train the agreement module
        # ----------------
        if self.train_mode == 'g':
            x1 = input_x[0]
            x1 = self.graph_net(x1)
            x2 = input_x[1]
            x2 = self.graph_net(x2)

            y_pred = self.gam_net(
                x1,
                x2
            )
            return y_pred

        elif self.train_mode == 'f':
            x1 = input_x[0]
            x1 = self.graph_net(x1)
            y_pred = self.clf_net(x1)
            return y_pred

# -----------------------
# Co-training
# -----------------------
def train_model(
        df
):
    global serialized_feature_col_list
    global feature_col_list

    import multiprocessing
    batch_size = 256
    num_epochs_g = 10
    num_epochs_f = 10
    num_proc = multiprocessing.cpu_count()
    lambda_LL = 0.1
    lambda_UL = 0.1
    lambda_UU = 0.1
    # GAM gets inputs as embeddings, which are obtained through the graph embeddings
    # that requires serialized feature ids
    g_feature_cols =  serialized_feature_col_list
    df_L = extract_labelled_df(df)
    net.train_mode = 'g'
    data_source_L1 = type1_Dataset(
        df_L,
        x_cols = g_feature_cols,
        y_col = label_col
    )

    dataLoader_obj_L1a = DataLoader(
            data_source_L1,
            batch_size = batch_size,
            shuffle=False,
            num_workers=num_proc,
            sampler=RandomSampler(data_source_L1)
        )
    dataLoader_obj_L1b = DataLoader(
        data_source_L1,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_proc,
        sampler=RandomSampler(data_source_L1)
    )

    optimizer_g = torch.optim.Adam(
        [net.graph_net.parameters(), net.gam_net.parameters()],
        lr=0.005
    )
    optimizer_f = torch.optim.Adam(
        [net.graph_net.parameters(), net.gam_net.parameters()],
        lr=0.005
    )

    # ----
    # input_x1,y2 : from Dataloader ( L )
    # input x2,y2 : from Dataloader ( L )
    # For every pair, so nest them
    # -----

    optimizer_g.zero_grad()
    for epoch in range(num_epochs_g):
        record_loss = []
        for i, data_i in enumerate(dataLoader_obj_L1a):
            x1 = data_i[0]
            y1 = data_i[1]
            for j, data_j in enumerate(dataLoader_obj_L1b):
                x2 = data_j[0]
                y2 = data_j[1]
                input_x = [x1,x2]
                true_agreement = np.array((y1 == y2)).astype(float)
                true_agreement = torch.FloatTensor(true_agreement)
                pred_agreement = net(input_x)
                loss = gam_loss(pred_agreement, true_agreement )
                loss.backward()
                optimizer_g.step()
                record_loss.append(float(loss))


    # -----------------------
    # Train the classifier
    # Use only labelled data
    # ----------------------

    net.train_mode = 'f'
    data_source_L2 = type1_Dataset(
        df_L,
        x_cols=g_feature_cols,
        y_col=label_col

    )
    dataLoader_obj_L2 = DataLoader(
        data_source_L2,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_proc,
        sampler=RandomSampler(data_source_L2)
    )
    optimizer_f.zero_grad()
    for epoch in range(num_epochs_f):

        for l_i, l_data in enumerate(dataLoader_obj_L2):
            x1 = l_data[0]
            y_true = l_data[1]
            pred_label = net(x1)
            loss = clf_loss(pred_label, y_true)
            loss.backward()
            optimizer_f.step()

            # LL


            # UL
        except StopIteration:
            continue



        # UU













# -------------------------------------------------- #
setup_config('us_import1')
