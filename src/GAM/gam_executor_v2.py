#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import copy
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
from time import time
from pandarallel import pandarallel
pandarallel.initialize()
import argparse
from datetime import datetime
import multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler, SequentialSampler
import yaml
import pickle
import torch
import torch.nn as nn
import logging
from torch import FloatTensor as FT
from torch import LongTensor as LT

try:
    from .gam_module import gam_net_v1 as gam_net
    from .gam_module import gam_loss
    from .clf_net import clf_net_v2 as clf_MLP
    from .clf_net import clf_loss as clf_loss
    from .record_node import graph_net_v2 as graph_net
    from .torch_data_loader import type1_Dataset
    from .torch_data_loader import dataGeneratorWrapper
    from . import train_utils
    from .torch_data_loader import pairDataGenerator
    from .torch_data_loader import singleDataGenerator
    from . import data_preprocess
    from .GAM_SS_module import SS_network
except:
    from gam_module import gam_net_v1 as gam_net
    from gam_module import gam_loss
    from clf_net import clf_net_v2 as clf_MLP
    from clf_net import clf_loss as clf_loss
    from record_node import graph_net_v2 as graph_net
    from torch_data_loader import type1_Dataset
    from torch_data_loader import pairDataGenerator
    from torch_data_loader import singleDataGenerator
    import train_utils
    import data_preprocess
    from GAM_SS_module import SS_network


DEVICE = None
try:
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    DEVICE = torch.device(dev)
    print('Set Device :: ', DEVICE)
    print('Cuda available ::', torch.cuda.is_available(),
          'Cuda current device ::', torch.cuda.current_device(),
          torch.cuda.get_device_name(0))
except:
    print('No CUDA')

try:
    from torch import has_cudnn

    if has_cudnn:
        torch.cudnn.benchmark = False
        print('Set cudnn benchmark to True')
except:
    pass

try:
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    DEVICE = torch.device(dev)
    print('Set Device :: ', DEVICE)
    print('Cuda available ::', torch.cuda.is_available(),
          'Cuda current device ::', torch.cuda.current_device(),
          torch.cuda.get_device_name(0))
except:
    print('No CUDA')

try:
    from torch import has_cudnn

    if has_cudnn:
        torch.cudnn.benchmark = False
        print('Set cudnn benchmark to True')
except:
    pass


# ==================================== #

config_file = 'config.yaml'
CONFIG = None
DATA_SOURCE_DIR_1 = None
DATA_SOURCE_DIR_2 = None
model_use_data_DIR = None
DIR = None
logger = None
Logging_Dir = '.'
domain_dims = None
score_col = 'score'
fraud_col = 'fraud'
anomaly_col = 'anomaly'
id_col = 'PanjivaRecordID'
label_col = 'y'
true_label_col = 'y_true'
node_emb_dim = 128
feature_col_list = []
serial_mapping_df = None
is_labelled_col = 'labelled'
matrix_node_emb_path = None
confidence_bound = 0.2
epochs_f = 0
epochs_g = 0
log_interval_f = 10
log_interval_g = 10
max_IC_iter = 5
clf_mlp_layer_dimesnions = []
gam_encoder_dimensions_mlp = []
batch_size_g = 128
batch_size_f = 128
batch_size_r = 128
F_classifier_type = None
clf_type = 'MLP'

# =================================================

def setup_config(_DIR):
    global CONFIG
    global config_file
    global DATA_SOURCE_DIR_1
    global DATA_SOURCE_DIR_2
    global DIR
    global Logging_Dir
    global model_use_data_DIR
    global domain_dims
    global feature_col_list
    global serial_mapping_df
    global serialized_feature_col_list
    global matrix_node_emb_path
    global confidence_bound
    global epochs_f
    global epochs_g
    global log_interval_f
    global log_interval_g
    global max_IC_iter
    global clf_mlp_layer_dimesnions
    global gam_encoder_dimensions_mlp
    global batch_size_g
    global batch_size_f
    global batch_size_r
    global clf_type

    if _DIR is not None:
        DIR = _DIR

    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)
    clf_type = CONFIG['clf_type']
    DATA_SOURCE_DIR_1 = CONFIG['DATA_SOURCE_DIR_1']
    DATA_SOURCE_DIR_2 = CONFIG['DATA_SOURCE_DIR_2']

    DATA_SOURCE_DIR_1 = os.path.join(DATA_SOURCE_DIR_1, DIR)
    DATA_SOURCE_DIR_2 = os.path.join(DATA_SOURCE_DIR_2, DIR)

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
    serialized_feature_col_list = ['_' + _ for _ in feature_col_list]
    serial_mapping_df_path = os.path.join(
        CONFIG['serial_mapping_df_loc'],
        DIR,
        CONFIG['serial_mapping_df_name']
    )
    serial_mapping_df = pd.read_csv(serial_mapping_df_path, index_col=None)
    matrix_node_emb_path = os.path.join(CONFIG['matrix_node_emb_loc'], DIR, CONFIG['matrix_node_emb_file'])
    confidence_bound = CONFIG['confidence_bound']
    epochs_g = CONFIG['epochs_g']
    epochs_f = CONFIG['epochs_f']
    log_interval_f = CONFIG['log_interval_f']
    log_interval_g = CONFIG['log_interval_g']
    max_IC_iter = CONFIG['max_IC_iter']
    clf_mlp_layer_dimesnions = [
        int(_)
        for _ in CONFIG['classifier_mlp_layers_1'].split(',')
    ]
    gam_encoder_dimensions_mlp = [
        int(_)
        for _ in CONFIG['gam_encoder_dimensions_mlp'].split(',')
    ]

    batch_size_g = CONFIG['batch_size_g']
    batch_size_f = CONFIG['batch_size_f']
    batch_size_r = CONFIG['batch_size_r']
    Logging_Dir = CONFIG['Logging_Dir']
    logger = get_logger()
    logger.info(str(datetime.utcnow()))
    return


def get_logger():
    global Logging_Dir
    global DIR
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    OP_DIR = os.path.join(Logging_Dir, DIR)
    log_file = 'results.log'
    if not os.path.exists(Logging_Dir):
        os.mkdir(Logging_Dir)

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    log_file_path = os.path.join(OP_DIR, log_file)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    return



df_target, normal_data_samples_df, features_F, features_G = data_preprocess.get_data_plus_features(
        DATA_SOURCE_DIR_1,
        DATA_SOURCE_DIR_2,
        model_use_data_DIR,
        clf_type,
        domain_dims,
        serial_mapping_df,
        score_col,
        is_labelled_col,
        label_col,
        true_label_col,
        fraud_col,
        anomaly_col
)

def read_matrix_node_emb (matrix_node_emb_path):
    emb = np.load(matrix_node_emb_path)
    return emb


matrix_node_emb = read_matrix_node_emb(matrix_node_emb_path)
node_emb_dim = matrix_node_emb.shape[-1]
num_domains = len(domain_dims)

# matrix_node_emb = FT(matrix_node_emb).to(DEVICE)
matrix_node_emb = FT(matrix_node_emb)

dict_clf_initilize_inputs = {
    'mlp_layer_dims' : clf_mlp_layer_dimesnions,
    'dropout' : 0.05,
    'activation':'relu'
}

NN = SS_network(
            DEVICE,
            node_emb_dimension=node_emb_dim,
            num_domains=num_domains,
            matrix_pretrained_node_embeddings=matrix_node_emb, # [Number of entities, embedding dimension]
            list_gam_encoder_dimensions = gam_encoder_dimensions_mlp,
            clf_type = clf_type,
            dict_clf_initilize_inputs = dict_clf_initilize_inputs
)




