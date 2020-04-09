#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import yaml
import pandas as pd
import numpy as np
import os
import argparse
import pickle
import sys
sys.path.append('./../..')
sys.path.append('./..')


try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from .src.data_fetcher import data_fetcher_v2 as data_fetcher



model_name = 'metapath2vec_gensim'
DIR = None
config_file = 'config.yaml'
model_use_data_DIR = None
randomWalk_DIR = None
SOURCE_DATA_DIR_1 = None
SOURCE_DATA_DIR_2 = None
metapath2vec_data_DIR = None

def set_up_config(_DIR = None):
    global CONFIG
    global config_file
    global DIR
    global model_use_data_DIR
    global serializedRandomWalk_DIR
    global randomWalk_DIR
    global model_name
    global model_weights_data
    global SOURCE_DATA_DIR_1
    global SOURCE_DATA_DIR_2
    global metapath2vec_data_DIR
    if _DIR is not None:
        DIR = _DIR

    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    SOURCE_DATA_DIR_1 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_1'], DIR
    )

    SOURCE_DATA_DIR_2 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_2'], DIR
    )

    model_use_data_DIR = CONFIG['model_use_data_DIR']
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    model_weights_data = CONFIG['model_weights_data']
    if not os.path.exists(model_weights_data):
        os.mkdir(model_weights_data)
    model_weights_data = os.path.join(
        model_weights_data ,DIR , model_name
    )
    model_data_dir = CONFIG['mp2v_g_data_dir']
    model_data_dir = os.path.join(model_use_data_DIR, model_data_dir)
    return

def get_domain_dims():
    global CONFIG
    global DIR
    return data_fetcher.get_domain_dims(CONFIG['SOURCE_DATA_DIR_1'], DIR)




