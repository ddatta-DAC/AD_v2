#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import yaml
import glob
import pandas as pd
import numpy as np
import os
import argparse
import pickle
import sys
sys.path.append('./../..')
sys.path.append('./..')

try:
    from .metapath2vec_g import metapath2vec_g_model
except:
    from metapath2vec_g import metapath2vec_g_model

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
mp2v_g_data_dir = None
text_data_file = None
model_save_loc= None
model_save_file= None
weights_np_save_path = None

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
    global mp2v_g_data_dir
    global text_data_file
    global RW_dir
    global model_save_loc
    global model_save_file
    global weights_np_save_path

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
    model_weights_data = os.path.join( model_weights_data , model_name)
    if not os.path.exists(model_weights_data):
        os.mkdir(model_weights_data)
    model_weights_data = os.path.join( model_weights_data , DIR )
    if not os.path.exists(model_weights_data):
        os.mkdir(model_weights_data)

    RW_dir = CONFIG['RW_Samples_DIR']
    RW_dir = os.path.join(model_use_data_DIR, RW_dir)

    mp2v_g_data_dir = CONFIG['mp2v_g_data_dir']
    mp2v_g_data_dir = os.path.join(model_use_data_DIR, mp2v_g_data_dir)
    if not os.path.exists(mp2v_g_data_dir):
        os.mkdir(mp2v_g_data_dir)
    text_data_file = os.path.join(mp2v_g_data_dir, 'gensim_corpus.txt')
    setup_data()

    model_save_path = CONFIG['model_weights_data']
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    model_save_path = os.path.join(model_save_path, model_name)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    model_save_loc = os.path.join(model_save_path,  DIR)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    model_save_file = 'mp2v_gensim.data'
    weights_np_save_path = os.path.join(model_save_loc, 'mp2v_gensim_weights.npy')
    print(weights_np_save_path, model_save_loc, os.path.exists(model_save_loc))


    return

def get_domain_dims():
    global CONFIG
    global DIR
    return data_fetcher.get_domain_dims(CONFIG['SOURCE_DATA_DIR_1'], DIR)

# --------------------------------------------------------- #

def setup_data():
    # Check if folder exists
    global model_data_dir
    global text_data_file
    global model_use_data_DIR
    global RW_dir

    if os.path.exists(text_data_file):
        print('Data file present')
        # return

    print(model_use_data_DIR)
    target_files = glob.glob(
        os.path.join(RW_dir,'**_walks.npy')
    )
    res = []
    for _file in target_files:
        np_arr = np.load(_file)
        res.extend(np_arr)

    res = np.array(res)

    np.savetxt(
        text_data_file,
        res,
        fmt ='%d',
        delimiter=' ',
        newline = '\n'
    )
    return

# ========================================================= #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import4', 'us_import5', 'us_import6'],
    default='us_import4'
)

args = parser.parse_args()
DIR = args.DIR
set_up_config(DIR)


model_obj = metapath2vec_g_model.get_model_obj(
    corpus_txt_file_path = text_data_file,
    emb_size=128,
    window_size = 2,
    model_save_path = os.path.join(model_save_loc,model_save_file),
    load_saved = False
)
domain_dims = get_domain_dims()
entity_count = sum(list(domain_dims.values()))
metapath2vec_g_model.save_weights(
        model_obj,
        entity_count,
        weights_np_save_path
)



