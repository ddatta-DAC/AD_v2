'''
# =========================================== #
# Generates the model data for APE
# =========================================== #
'''

import pickle
import pandas as pd
import numpy as np
import os
import yaml
import argparse
import re
import glob

# ----------------------------------- #

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


def set_up_config(_DIR):
    global CONFIG_FILE
    global CONFIG
    global use_cols
    global freq_bound
    global num_neg_samples_ape
    global DIR
    global save_dir
    global column_value_filters
    global num_neg_samples_v1

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

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
    return


# =========================================== #
# create .npy pickles for test data
# =========================================== #
def create_base_test_matrices(
        save_dir,
        id_col,
):
    test_data_file = os.path.join(save_dir, 'test_data.csv')
    test_df = pd.read_csv(
        test_data_file,
        index_col=None
    )

    # Read in domain_dims, domains are sorted
    with open(os.path.join(save_dir, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)
    domains = list(domain_dims.keys())

    test_idList = list(test_df[id_col])
    del test_df[id_col]

    test_x = test_df.values
    # test_matrix_x_positive.npy
    fpath = os.path.join(save_dir, 'test_matrix_x_positive.npy')
    np.save(fpath, test_x)

    fpath = os.path.join(save_dir, 'test_idList.npy')
    np.save(fpath, test_idList)

    return


# =========================================== #
# create .npy pickles for anomalies data
# =========================================== #

def create_base_anomaly_matrices(
        save_dir,
        id_col
):
    # ------------------------
    # find the anomaly files !
    # ------------------------

    csv_file_list = sorted(glob.glob(os.path.join(save_dir, 'anomalies_type**.csv')))

    for _file in csv_file_list:
        _df = pd.read_csv(
            _file,
            index_col=None
        )
        anomaly_type = int(re.findall(r'\d+', _file)[-1])

        id_list = list(_df[id_col])
        del _df[id_col]

        x = _df.values

        fpath = os.path.join(save_dir, 'matrix_anomaly_x_type' + str(anomaly_type) + '.npy')
        np.save(fpath, x)
        fpath = os.path.join(save_dir, 'matrix_anomaly_idList_type' + str(anomaly_type) + '.npy')
        np.save(fpath, id_list)

    return


# =========================================== #
# APE specific stuff
# =========================================== #

def create_ape_model_data(
        term_2_col,
        term_4_col,
        save_dir,
        id_col,
        ns_id_col
):
    train_neg_data_file = os.path.join(save_dir, 'negative_samples_ape_1.csv')
    train_pos_data_file = os.path.join(save_dir, 'train_data.csv')

    train_pos_df = pd.read_csv(
        train_pos_data_file
    )

    neg_samples_df = pd.read_csv(
        train_neg_data_file,
        index_col=0
    )

    # ------------------- #

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
        vals_n = vals_n.astype('int32')
        vals_p = list(row.values.astype('int32'))
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

    print('Shapes : matrix_pos ', matrix_pos.shape, ' matrix_neg ', matrix_neg.shape)
    print('term 2', term_2.shape, 'term 4', term_4.shape)

    # Save files
    f_path = os.path.join(save_dir, 'train_matrix_x_positive.npy')
    np.save(f_path, matrix_pos)

    f_path = os.path.join(save_dir, 'negative_samples_ape.npy')
    np.save(f_path, matrix_neg)

    f_path = os.path.join(save_dir, 'ape_term_2.npy')
    np.save(f_path, term_2)

    f_path = os.path.join(save_dir, 'ape_term_4.npy')
    np.save(f_path, term_4)

    return


# =========================================== #
# General model stuff
# meant for mead, but can be used elsewhere
# =========================================== #


def create_mead_model_data(
        save_dir,
        id_col,
        ns_id_col
):
    train_pos_data_file = os.path.join(save_dir, 'train_data.csv')
    train_neg_data_file = os.path.join(save_dir, 'negative_samples_v1.csv')

    # ------------------- #

    train_pos_df = pd.read_csv(
        train_pos_data_file,
        index_col=None
    )

    neg_samples_df = pd.read_csv(
        train_neg_data_file,
        index_col=0
    )

    feature_cols = list(train_pos_df.columns)
    feature_cols.remove(id_col)

    matrix_pos = []
    matrix_neg = []

    index = 0
    for i, row in train_pos_df.iterrows():
        _row = pd.Series(row, copy=True)
        _tmp = pd.DataFrame(
            neg_samples_df.loc[neg_samples_df[id_col] == row[id_col]],
            copy=True
        )

        del _tmp[ns_id_col]
        del _tmp[id_col]
        del _row[id_col]

        vals_n = np.array(_tmp.values)
        vals_p = list(_row.values)
        matrix_neg.append(vals_n)
        matrix_pos.append(vals_p)

        index += 1

    matrix_pos = np.array(matrix_pos)
    matrix_neg = np.array(matrix_neg)

    print(matrix_pos.shape, matrix_neg.shape)

    # Save files
    f_path = os.path.join(save_dir, 'train_matrix_x_positive.npy')
    np.save(f_path, matrix_pos)

    f_path = os.path.join(save_dir, 'negative_samples_mead.npy')
    np.save(f_path, matrix_neg)


# ========================================================= #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'china_export1', 'china_import1'],
    default=None
)

args = parser.parse_args()
DIR = args.DIR
set_up_config(DIR)

create_base_test_matrices(
    save_dir,
    id_col,
)

create_base_anomaly_matrices(
    save_dir,
    id_col
)

create_ape_model_data(
    term_2_col=term_2_col,
    term_4_col=term_4_col,
    save_dir=save_dir,
    id_col=id_col,
    ns_id_col=ns_id_col
)

create_mead_model_data(
    save_dir=save_dir,
    id_col=id_col,
    ns_id_col=ns_id_col
)
