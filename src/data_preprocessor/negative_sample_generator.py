import os
import sys
import pandas as pd
import numpy as np
import sklearn
import glob
import pickle
import random
from joblib import Parallel, delayed
import yaml
import math
import multiprocessing as mp
import argparse

try:
    from . import utils_createAnomalies as utils_local
except:
    import utils_createAnomalies as utils_local

# ========================================================= #

CONFIG_FILE = 'config_preprocessor_v02.yaml'
id_col = 'PanjivaRecordID'
ns_id_col = 'NegSampleID'
term_2_col = 'term_2'
term_4_col = 'term_4'
num_neg_samples_ape = None
use_cols = None
freq_bound = None
column_value_filters = None
num_neg_samples_v1 = None
save_dir = None
DIR = None
CONFIG = None
num_jobs = None

def set_up_config():
    global CONFIG_FILE
    global CONFIG
    global use_cols
    global freq_bound
    global num_neg_samples_ape
    global DIR
    global save_dir
    global column_value_filters
    global num_neg_samples_v1
    global num_jobs

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    DIR = CONFIG['DIR']
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = CONFIG[DIR]['use_cols']
    freq_bound = CONFIG[DIR]['low_freq_bound']
    num_neg_samples_ape = CONFIG[DIR]['num_neg_samples_ape']

    column_value_filters = CONFIG[DIR]['column_value_filters']
    num_neg_samples_v1 = CONFIG[DIR]['num_neg_samples']

    num_jobs = CONFIG['num_chunks']
    num_jobs = max(mp.cpu_count(), num_jobs)

    return


def get_neg_sample_ape(
        _k,
        column_id,
        column_name,
        ref_df,
        column_valid_values,
        orig_row,
        P_A,
        feature_cols_id
):
    global id_col
    global ns_id_col
    global term_4_col
    global term_2_col

    ref_df = pd.DataFrame(ref_df, copy=True)
    ref_df = utils_local.add_hash(
        ref_df,
        id_col
    )

    Pid_val = orig_row[id_col]
    check_duplicate = False

    while True:
        new_row = pd.Series(orig_row, copy=True)
        _random = random.sample(column_valid_values[column_name], 1)[0]
        new_row[column_name] = _random
        # Check is not a duplicate of something in training set

        new_row_hash = utils_local.get_hash_aux(new_row, id_col)
        if check_duplicate and utils_local.is_duplicate(ref_df, new_row_hash):
            continue

        new_row[ns_id_col] = int('10' + str(_k) + str(column_id) + str(Pid_val) + '01')
        new_row[term_4_col] = np.log(P_A[column_id][_random])

        _tmp = 0
        for _fci, _fcn in feature_cols_id.items():
            _val = P_A[_fci][orig_row[_fcn]]
            _tmp += math.log(_val, math.e)
        _tmp /= len(feature_cols_id)
        new_row[term_2_col] = _tmp
        print(" generated new row  ::  Pid_val :: {} get_neg_sample_ape ".format(Pid_val))
    return   new_row


def create_negative_samples_ape_aux(
        idx,
        df_chunk,
        feature_cols,
        ref_df,
        column_valid_values,
        save_dir,
        P_A
):
    global ns_id_col
    global term_4_col
    global term_2_col
    global id_col
    global num_neg_samples_ape
    print(' IN ::  create_negative_samples_ape_aux ')
    ns_id_col = 'NegSampleID'
    term_2_col = 'term_2'
    term_4_col = 'term_4'

    feature_cols_id = {
        e[0]: e[1]
        for e in enumerate(feature_cols)
    }

    new_df = pd.DataFrame(
        columns=list(ref_df.columns)
    )

    new_df[ns_id_col] = 0
    new_df[term_4_col] = 0
    new_df[term_2_col] = 0

    for i, row in df_chunk.iterrows():
        for column_id, column_name in feature_cols_id.items():
            for _k in range(num_neg_samples_ape):
                _res = get_neg_sample_ape(
                    _k, column_id,
                    column_name,
                    ref_df,
                    column_valid_values,
                    row,
                    P_A,
                    feature_cols_id
                )
                new_df = new_df.append(
                    _res,
                    ignore_index=True
                )

    if not os.path.exists(os.path.join(save_dir, 'tmp')):
        os.mkdir(os.path.join(save_dir, 'tmp'))
    f_name = os.path.join(save_dir, 'tmp', 'tmp_df_' + str(idx) + '.csv')
    new_df.to_csv(
        f_name,
        index=None
    )

    return f_name


def create_negative_samples_ape():
    global DIR
    global save_dir
    global id_col
    global ns_id_col
    global num_neg_samples_ape
    global CONFIG
    global num_jobs

    train_data_file = os.path.join(save_dir, CONFIG['train_data_file'])
    train_df = pd.read_csv(
        train_data_file,
        index_col=None
    )

    '''
    Randomly generate samples
    choose k=3 * m=8 = 24 negative samples per training instance
    For negative samples pick one entity & replace it it randomly 
    Validate if generated negative sample is not part of the test or training set
    '''

    ref_df = pd.DataFrame(
        train_df,
        copy=True
    )

    feature_cols = list(train_df.columns)
    feature_cols.remove(id_col)
    feature_cols_id = {
        e[0]: e[1]
        for e in enumerate(feature_cols)
    }

    # get the domain dimensions
    with open(os.path.join(save_dir, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)

    print(' domain dimensions :: ', domain_dims)

    # This id for the 4th term
    P_A = {}
    for _fci, _fcn in feature_cols_id.items():
        _series = pd.Series(train_df[_fcn])
        tmp = _series.value_counts(normalize=True)
        P_Aa = tmp.to_dict()
        for _z in range(domain_dims[_fcn]):
            if _z not in P_Aa.keys():
                P_Aa[_z] = math.pow(10, -3)
        P_A[_fci] = P_Aa

    # Store what are valid values for each columns
    column_valid_values = {}
    for _fc_name in feature_cols:
        column_valid_values[_fc_name] = list(
            set(list(ref_df[_fc_name]))
        )

    list_df_chunks = utils_local.chunk_df(
        train_df,
        num_jobs
    )

    results = Parallel(n_jobs = num_jobs)(
        delayed(create_negative_samples_ape_aux)(
            _i,
            list_df_chunks[_i],
            feature_cols,
            ref_df,
            column_valid_values,
            save_dir,
            P_A
        )
        for _i in range(len(list_df_chunks))
    )
    results = sorted(results)

    new_df = None
    for _f in results:
        _df = pd.read_csv(_f, index_col=None)
        if new_df is None:
            new_df = _df
        else:
            new_df = new_df.append(_df, ignore_index=True)
        print(' >> ', len(new_df))

    new_df.to_csv(os.path.join(save_dir, 'negative_samples_ape_1.csv'), index=False)
    return new_df

# ---------------------------------------------------- #

def create_ape_model_data():
    global DIR
    global term_2_col
    global term_4_col
    global save_dir
    global id_col
    global ns_id_col
    global num_neg_samples_ape

    train_pos_data_file = os.path.join(save_dir, 'train_data.csv')
    train_neg_data_file = os.path.join(save_dir, 'negative_samples_ape.csv')

    # ------------------- #

    train_pos_df = pd.read_csv(
        train_pos_data_file,
        index_col=None
    )

    neg_samples_df = pd.read_csv(
        train_neg_data_file,
        index_col=None
    )

    feature_cols = list(train_pos_df.columns)
    feature_cols.remove(id_col)

    matrix_pos = []
    matrix_neg = []

    term_2 = []
    term_4 = []

    index = 0
    for i, row in train_pos_df.iterrows():
        _tmp = pd.DataFrame(
            neg_samples_df.loc[neg_samples_df[id_col] == row[id_col]],
            copy=True
        )

        _term_2 = list(_tmp[term_2_col])[0]
        _term_4 = list(_tmp[term_4_col])

        del _tmp[ns_id_col]
        del _tmp[id_col]
        del _tmp[term_2_col]
        del _tmp[term_4_col]
        del row[id_col]

        vals_n = np.array(_tmp.values)
        vals_p = list(row.values)

        matrix_neg.append(vals_n)
        matrix_pos.append(vals_p)
        term_2.append(_term_2)
        term_4.append(_term_4)
        index += 1

    matrix_pos = np.array(matrix_pos)
    matrix_neg = np.array(matrix_neg)

    matrix_pos = matrix_pos.astype(np.int32)
    matrix_neg = matrix_neg.astype(np.int32)

    term_2 = np.array(term_2)
    term_4 = np.array(term_4)

    print(matrix_pos.shape, matrix_neg.shape)
    print(term_2.shape, term_4.shape)

    # Save files

    f_path = os.path.join(save_dir, 'ape_negative_samples.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            matrix_neg,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    f_path = os.path.join(save_dir, 'ape_term_2.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            term_2,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    f_path = os.path.join(save_dir, 'ape_term_4.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            term_4,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    return


'''
Negative sample generation for the new  model
based on the concept 1 - feature bagging
'''


def get_neg_sample_v1(
        _k,
        ref_df,
        column_valid_values,
        orig_row,
        feature_cols_id
):
    global id_col
    global ns_id_col

    Pid_val = orig_row[id_col]
    num_features = len(feature_cols_id)
    num_randomizations = random.randint(1, int(num_features / 2))

    # iterate while a real noise is not generated
    while True:
        target_cols = [feature_cols_id[_]
                       for _ in random.sample(
                list(feature_cols_id.keys()),
                k=num_randomizations
            )
                       ]
        c_vals = {}
        for _tc in target_cols:
            c_vals[_tc] = random.sample(column_valid_values[_tc], 1)[0]

        new_row = pd.Series(orig_row, copy=True)
        for _col, _item_id in c_vals.items():
            new_row[_col] = _item_id

        _hash = utils_local.get_hash_aux(new_row, id_col)
        if not utils_local.is_duplicate(ref_df, _hash):
            new_row[ns_id_col] = int(str(Pid_val) + '01' + str(_k))
            break

    return new_row


def create_negative_samples_v1_aux(
        idx,
        df_chunk,
        feature_cols,
        ref_df,
        column_valid_values,
        save_dir
):
    global ns_id_col
    global id_col
    global num_neg_samples_v1

    ns_id_col = 'NegSampleID'
    feature_cols_id = {
        e[0]: e[1]
        for e in enumerate(feature_cols)
    }

    new_df = pd.DataFrame(
        columns=list(ref_df.columns)
    )

    new_df[ns_id_col] = 0
    for i, row in df_chunk.iterrows():

        for _k in range(num_neg_samples_v1):
            _res = get_neg_sample_v1(
                _k, ref_df, column_valid_values, row, feature_cols_id
            )
            new_df = new_df.append(
                _res,
                ignore_index=True
            )

    if not os.path.exists(os.path.join(save_dir, 'tmp')):
        os.mkdir(os.path.join(save_dir, 'tmp'))
    f_name = os.path.join(save_dir, 'tmp', 'tmp_df_' + str(idx) + '.csv')
    new_df.to_csv(
        f_name,
        index=None
    )
    return f_name


def create_negative_samples_v1():
    global DIR
    global save_dir
    global id_col
    global ns_id_col
    global num_neg_samples_v1
    global num_jobs

    train_data_file = os.path.join(save_dir, 'train_data.csv')

    train_df = pd.read_csv(
        train_data_file,
        index_col=None
    )

    '''
    Randomly generate samples
    choose 15 negative samples per training instance
    For negative samples pick m entities & replace it it randomly 
    m randomly between (1, d/2)
    Validate if generated negative sample is not part of the test or training set
    '''

    ref_df = pd.DataFrame(
        train_df,
        copy=True
    )

    feature_cols = list(train_df.columns)
    feature_cols.remove(id_col)

    # feature_cols_id = {
    #     e[0]: e[1]
    #     for e in enumerate(feature_cols)
    # }
    # # get the domain dimensions
    # with open(
    #         os.path.join(save_dir, 'domain_dims.pkl'), 'rb'
    # ) as fh:
    #     domain_dims = pickle.load(fh)

    # Store what are valid values for each columns
    column_valid_values = {}
    for _fc_name in feature_cols:
        column_valid_values[_fc_name] = list(set(list(ref_df[_fc_name])))

    chunk_len = int(len(train_df) / (num_jobs - 1))

    list_df_chunks = np.split(
        train_df.head(
            chunk_len * (num_jobs - 1)
        ), num_jobs - 1
    )

    end_len = len(train_df) - chunk_len * (num_jobs - 1)
    list_df_chunks.append(train_df.tail(end_len))
    for _l in range(len(list_df_chunks)):
        print(len(list_df_chunks[_l]), _l)


    results = Parallel(n_jobs=num_jobs)(
        delayed
        (create_negative_samples_v1_aux)(
            _i,
            list_df_chunks[_i],
            feature_cols,
            ref_df,
            column_valid_values,
            save_dir
        )
        for _i in range(len(list_df_chunks))
    )

    new_df = None
    results = sorted(results)

    for _f in results:
        _df = pd.read_csv(_f, index_col=None)

        if new_df is None:
            new_df = _df
        else:
            new_df = new_df.append(_df, ignore_index=True)
        print(' >> ', len(new_df))

    new_df.to_csv(os.path.join(save_dir, 'negative_samples_v1.csv'), index=False)
    return new_df


# ========================================================= #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default=None
)
# ========================================================= #

args = parser.parse_args()
DIR = args.DIR
set_up_config(DIR)

# ======================================= #

set_up_config()
# create_negative_samples_ape()
# create_ape_model_data()

create_negative_samples_v1()
# ======================================= #
