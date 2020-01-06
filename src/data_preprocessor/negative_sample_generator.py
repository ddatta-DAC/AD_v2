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


def set_up_config():
    global CONFIG_FILE
    global use_cols
    global freq_bound
    global num_neg_samples_ape
    global DIR
    global save_dir
    global column_value_filters
    global num_neg_samples_v1

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    DIR = CONFIG['_DIR']
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = CONFIG[DIR]['use_cols']
    freq_bound = CONFIG[DIR]['low_freq_bound']
    num_neg_samples_ape = CONFIG[DIR]['num_neg_samples_ape']
    freq_bound = CONFIG[DIR]['low_freq_bound']
    column_value_filters = CONFIG[DIR]['column_value_filters']
    num_neg_samples_v1 = CONFIG[DIR]['num_neg_samples_v1']

    return




def get_neg_sample_ape(_k, column_id, column_name, ref_df, column_valid_values, orig_row, P_A, feature_cols_id):
    global id_col
    global ns_id_col
    global term_4_col
    global term_2_col


    Pid_val = orig_row[id_col]
    while True:
        new_row = pd.Series(orig_row, copy=True)
        _random = random.sample(
            column_valid_values[column_name], 1
        )[0]
        new_row[column_name] = _random
        if validate(new_row, ref_df):

            new_row[ns_id_col] = int('10' + str(_k) + str(column_id) + str(Pid_val) + '01')
            new_row[term_4_col] = np.log(P_A[column_id][_random])
            _tmp = 0
            for _fci, _fcn in feature_cols_id.items():
                _val = P_A[_fci][orig_row[_fcn]]
                _tmp += math.log(_val, math.e)
            _tmp /= len(feature_cols_id)
            new_row[term_2_col] = _tmp
            return new_row


def create_negative_samples_ape_aux(
        idx, df_chunk, feature_cols, ref_df, column_valid_values, save_dir, P_A):
    global ns_id_col
    global term_4_col
    global term_2_col
    global id_col
    global num_neg_samples_ape

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
                    _k, column_id, column_name, ref_df, column_valid_values, row, P_A, feature_cols_id
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

    num_chunks = 40

    train_data_file = os.path.join(save_dir, 'train_data.csv')

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

    chunk_len = int(len(train_df) / (num_chunks - 1))

    list_df_chunks = np.split(
        train_df.head(
            chunk_len * (num_chunks - 1)
        ), num_chunks - 1
    )

    end_len = len(train_df) - chunk_len * (num_chunks - 1)
    list_df_chunks.append(train_df.tail(end_len))

    for _l in range(len(list_df_chunks)):
        print(' Length of chunk ', _l, ' :: ', len(list_df_chunks[_l]))

    results = Parallel(n_jobs=num_chunks)(
        delayed(create_negative_samples_ape_aux)(
            _i, list_df_chunks[_i], feature_cols, ref_df, column_valid_values, save_dir, P_A)
        for _i in range(len(list_df_chunks))
    )

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

def create_ape_model_data():
    global DIR
    global term_2_col
    global term_4_col
    global save_dir
    global id_col
    global ns_id_col
    global num_neg_samples_ape

    train_pos_data_file = os.path.join(save_dir, 'train_data.csv')
    test_data_file = os.path.join(save_dir, 'test_data_v1.csv')
    train_neg_data_file = os.path.join(save_dir, 'negative_samples_ape.csv')

    # ------------------- #

    train_pos_df = pd.read_csv(
        train_pos_data_file,
        index_col=None
    )

    test_df = pd.read_csv(
        test_data_file,
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