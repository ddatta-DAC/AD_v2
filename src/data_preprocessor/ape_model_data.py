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

def create_ape_model_data(
    term_2_col,
    term_4_col,
    save_dir,
    id_col,
    ns_id_col
):
    print('hello ')
    return

    train_pos_data_file = os.path.join(save_dir, 'train_data.csv')
    train_neg_data_file = os.path.join(save_dir, 'negative_samples_ape_1.csv')
    test_data_file = os.path.join(save_dir, 'test_data.csv')

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

    matrix_test = test_df.values

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
    f_path = os.path.join(save_dir, 'matrix_train_positive.npy')
    np.save(f_path, matrix_pos)

    f_path = os.path.join(save_dir,'ape_negative_samples.npy')
    np.save(f_path , matrix_neg)

    f_path = os.path.join(save_dir, 'ape_term_2.npy')
    np.save(f_path, term_2)

    f_path = os.path.join(save_dir, 'ape_term_4.npy')
    np.save(f_path, term_4)

    f_path = os.path.join(save_dir, 'matrix_test_positive.npy')
    np.save(f_path , matrix_test)



# ========================================================= #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'china_export1', 'china_import1'],
    default=None
)

args = parser.parse_args()
DIR = args.DIR
set_up_config(DIR)

create_ape_model_data(
    term_2_col = term_2_col,
    term_4_col = term_4_col,
    save_dir = save_dir,
    id_col= id_col,
    ns_id_col = ns_id_col
)