#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import numpy as np
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')
import glob
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from pandarallel import pandarallel
pandarallel.initialize()
import yaml
import argparse


CONFIG = None
CONFIG_FILE = 'config.yaml'
DATA_SOURCE_DIR_1 = None
model_use_data_DIR = None
RW_dir = None
Serialized_RW_dir = None
SAVE_DIR_loc = None
domain_dims = None
metapath2vec_data_DIR = None
mp2v_context_size = None

def set_up_config(_DIR = None):
    global CONFIG
    global CONFIG_FILE
    global DATA_SOURCE_DIR_1
    global DIR
    global SAVE_DIR_loc
    global Refresh
    global Serialized_RW_dir
    global RW_dir
    global domain_dims
    global flag_REFRESH_create_mp2v_data
    global model_use_data_DIR
    global mp2v_context_size
    global metapath2vec_data_DIR


    if _DIR is not None:
        DIR = _DIR

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    DATA_SOURCE_DIR_1 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_1'], DIR
    )

    model_use_data_DIR = os.path.join(CONFIG['model_use_data_DIR'], DIR)

    RW_dir = CONFIG['RW_Samples_DIR']
    RW_dir = os.path.join(model_use_data_DIR, RW_dir)
    metapath2vec_data_DIR = CONFIG['mp2v_data']
    metapath2vec_data_DIR = os.path.join(model_use_data_DIR, metapath2vec_data_DIR)

    if not os.path.exists(metapath2vec_data_DIR):
        os.mkdir(metapath2vec_data_DIR)

    mp2v_context_size = CONFIG['mp2v_context_size']
    return

# -----------------------------------------
# Create ingestion data for metapath2vec model
# ---------------------------------------------------------- #
# Function to create data specific to metapath2vec_1 model
# Following the skip-gram, a word and its context are chosen as well as corresponding negative 'context'
# ---------------------------------------------------------- #


def create_data_aux(args) :
    _pos_arr = args[0]
    _neg_samples_arr = args[1]
    _ctxt_size = args[2]
    _pos_arr = np.reshape(_pos_arr,-1)
    k = _ctxt_size//2
    arr_len = _pos_arr.shape[-1]
    centre = []
    context = []
    neg_samples = []

    # from i-k to i+1
    del_pos = k
    for i in range( k, arr_len-k ):
        cur_pos = _pos_arr[i-k:i+k+1]
        cur_centre_val = cur_pos[k]
        centre.append(cur_centre_val)
        # remove the central "word" from cur_pos
        ctx = np.delete(cur_pos, del_pos)
        context.append(list(ctx))
        cur_ns = _neg_samples_arr[:, i-k:i+k+1]
        cur_ns = np.delete(cur_ns, del_pos, axis=1)
        neg_samples.append(cur_ns)

    context = np.array(context)
    neg_samples= np.array(neg_samples)
    return (centre, context, neg_samples)


def create_metapath2vec_ingestion_data(
    source_dir = None,
    target_dir = None,
    ctxt_size = 6
):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    print(source_dir)
    _files = glob.glob(
        source_dir + '/../**.npy'
    )

    mp_specs = sorted(set([ _.split('/')[-1].split('.')[0] for _ in _files]))
    res_centre = []
    res_context = []
    res_neg_samples = []

    for mp_spec in mp_specs :
        pos_samples_file = os.path.join(
            source_dir, mp_spec+'_walks.npy'
        )
        neg_samples_file = os.path.join(
            source_dir, mp_spec+'_neg_samples.npy'
        )
        pos_samples = np.load(pos_samples_file)
        neg_samples = np.load(neg_samples_file)

        num_jobs = multiprocessing.cpu_count()
        count = pos_samples.shape[0]
        results = Parallel( num_jobs )(
            delayed(create_data_aux)(
                (pos_samples[i], neg_samples[i], ctxt_size),)
            for i in range(count)
        )

        for _result in results :
            _centre = _result[0]
            _context = _result[1]
            _neg_samples = _result[2]


            res_centre.extend(_centre)
            res_context.extend(_context)
            res_neg_samples.extend(_neg_samples)

    centre = np.array(res_centre)
    context = np.array(res_context)
    neg_samples = np.array(res_neg_samples)

    print(centre.shape , context.shape , neg_samples.shape)
    # -----------------
    # Save data
    # -----------------
    np.save(
        os.path.join(target_dir, 'x_target.npy'),
        centre
    )
    np.save(
        os.path.join(target_dir, 'x_context.npy'),
        context
    )
    np.save(
        os.path.join(target_dir, 'x_neg_samples.npy'),
        neg_samples
    )
    return

# ------------------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)

args = parser.parse_args()
set_up_config(args.DIR)


mp2v_data_loc = create_metapath2vec_ingestion_data(
    source_dir = RW_dir,
    target_dir = metapath2vec_data_DIR,
    ctxt_size = mp2v_context_size
)
