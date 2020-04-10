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
import re
from pandarallel import pandarallel
from collections import Counter
import argparse

pandarallel.initialize(progress_bar=True)
sys.path.append('.')
sys.path.append('./..')
try:
    import utils_preprocess
except:
    from . import utils_preprocess

# -------------------------- #

DATA_SOURCE = None
DIR_LOC = None
CONFIG = None
CONFIG_FILE = 'config_preprocessor_v02.yaml'
id_col = 'PanjivaRecordID'
ns_id_col = 'NegSampleID'
num_neg_samples_ape = None
use_cols = None
freq_bound = None
column_value_filters = None
num_neg_samples = None
save_dir = None
cleaned_csv_subdir = None


# -------------------------- #

def get_regex(_type):
    global DIR

    if DIR == 'us_import1':
        if _type == 'train':
            return '.*0[1-3]_2015.csv'
        if _type == 'test':
            return '.*((04)|(05))_2015.csv'

    if DIR == 'us_import2':
        if _type == 'train':
            return '.*(09|10|11|12)_2015.csv'
        if _type == 'test':
            return '.*0[1-4]_2016.csv'

    if DIR == 'us_import3':
        if _type == 'train':
            return '.*(0[2-5]_2016).csv'
        if _type == 'test':
            return '.*0[6-9]_2016.csv'

    return '*.csv'


def get_files(DIR, _type='all'):
    global DATA_SOURCE
    data_dir = DATA_SOURCE

    regex = get_regex(_type)
    c = glob.glob(os.path.join(data_dir, '*'))

    def glob_re(pattern, strings):
        return filter(re.compile(pattern).match, strings)

    files = sorted([_ for _ in glob_re(regex, c)])

    print('DIR ::', DIR, ' Type ::', _type, 'Files count::', len(files))
    return files


def set_up_config(_DIR = None):
    global DIR
    global CONFIG
    global CONFIG_FILE
    global use_cols
    global freq_bound
    global num_neg_samples_ape
    global save_dir
    global column_value_filters
    global num_neg_samples
    global cleaned_csv_subdir
    global DATA_SOURCE
    global DIR_LOC

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is not None:
        DIR = _DIR
        CONFIG['DIR'] = _DIR
    else:
        DIR = CONFIG['DIR']

    DIR_LOC = re.sub('[0-9]', '', DIR)
    DATA_SOURCE = os.path.join('./../../Data_Raw', DIR_LOC)
    save_dir =  CONFIG['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    cleaned_csv_subdir = os.path.join(
        save_dir,
        CONFIG['cleaned_csv_subdir']
    )

    if not os.path.exists(cleaned_csv_subdir):
        os.mkdir(cleaned_csv_subdir)

    use_cols = CONFIG[DIR]['use_cols']
    freq_bound = CONFIG[DIR]['low_freq_bound']
    column_value_filters = CONFIG[DIR]['column_value_filters']
    num_neg_samples_ape = CONFIG[DIR]['num_neg_samples_ape']
    num_neg_samples = CONFIG[DIR]['num_neg_samples']


'''
Replace attribute with id specific to a domain
'''


def replace_attr_with_id(row, attr, val2id_dict):
    val = row[attr]
    if val not in val2id_dict.keys():
        print(attr, val)
        return None
    else:
        return val2id_dict[val]


'''
Converts the train df to ids 
Returns :
col_val2id_dict  { 'col_name': { 'col1_val1': id1,  ... } , ... }
'''


def convert_to_ids(
        df,
        save_dir
):
    global id_col
    global freq_bound
    print('freq_bound ==', freq_bound)

    feature_columns = list(df.columns)
    feature_columns.remove(id_col)
    feature_columns = list(sorted(feature_columns))
    dict_DomainDims = {}
    col_val2id_dict = {}

    for col in sorted(feature_columns):
        vals = list(set(df[col]))
        vals = list(sorted(vals))

        id2val_dict = {
            e[0]: e[1]
            for e in enumerate(vals, 0)
        }
        print(id2val_dict)

        val2id_dict = {
            v: k for k, v in id2val_dict.items()
        }
        col_val2id_dict[col] = val2id_dict


        # Replace
        df[col] = df.parallel_apply(
            replace_attr_with_id,
            axis=1,
            args=(
                col,
                val2id_dict,
            )
        )

        dict_DomainDims[col] = len(id2val_dict)

    print(' Feature columns :: ', feature_columns)
    print('dict_DomainDims ', dict_DomainDims)

    # -------------
    # Save the domain dimensions
    # -------------

    file = 'domain_dims.pkl'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            dict_DomainDims,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    file = 'col_val2id_dict.pkl'
    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            col_val2id_dict,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    return df, col_val2id_dict


'''
Join the csv files into 1 single Dataframe
Removes missing values
Input : file_path_list
'''

'''
Remove the rows with entities that have very low frequency.
'''


def remove_low_frequency_values(df):
    global id_col
    global freq_bound

    freq_column_value_filters = {}
    feature_cols = list(df.columns)
    feature_cols.remove(id_col)
    # ----
    # figure out which entities are to be removed
    # ----
    for c in feature_cols:
        values = list(df[c])
        freq_column_value_filters[c] = []
        obj_counter = Counter(values)

        for _item, _count in obj_counter.items():
            if _count < freq_bound:
                freq_column_value_filters[c].append(_item)

    print('Removing :: ')
    for c, _items in freq_column_value_filters.items():
        print('column : ', c, 'count', len(_items))

    print(' DF length : ', len(df))
    for col, val in freq_column_value_filters.items():
        df = df.loc[~df[col].isin(val)]

    print(' DF length : ', len(df))
    return df


def HSCode_cleanup(list_df, DIR_LOC, config):
    hscode_col = config['hscode_col']
    # ----- #
    # Expert curated HS codes
    hs_code_filter_file = os.path.join(
        config['hscode_filter_file_loc'],
        DIR_LOC + config['hscode_filter_file_pattern']
    )

    tmp = pd.read_csv(
        hs_code_filter_file,
        index_col=None,
        header=None
    )
    target_codes = list(tmp[0])

    def hsc_proc(_code):
        return str(_code)[:4]

    target_codes = list(sorted([hsc_proc(_) for _ in target_codes]))

    def filter_by_ExpertHSCodeList(_code, target_codes):
        if _code[:2] in target_codes or _code[:4] in target_codes:
            return _code
        return None

    def remove_dot(_code):
        return _code.replace('.', '')

    list_processed_df = []

    for df in list_df:
        df = df.dropna()
        df[hscode_col] = df[hscode_col].astype(str)

        df[hscode_col] = df[hscode_col].parallel_apply(
            remove_dot
        )

        df[hscode_col] = df[hscode_col].parallel_apply(
            filter_by_ExpertHSCodeList,
            args=(target_codes,)
        )

        df = df.dropna()
        df = lexical_sort_cols(df, id_col)

        if df is not None and len(df) > 0:
            df = df.dropna()
            list_processed_df.append(df)

    # --------- #
    print([len(_) for _ in list_processed_df])
    return list_processed_df


'''
Apply :: column_value_filters
Remove values which are garbage & known to us
'''


def apply_value_filters(list_df):
    global column_value_filters

    if type(column_value_filters) != bool:
        list_processed_df = []
        for df in list_df:
            for col, val in column_value_filters.items():
                df = df.loc[~df[col].isin(val)]
            list_processed_df.append(df)
        return list_processed_df
    return list_df


'''
Check if a row is not present in reference_DF
reference df should have a hash
'''


def ensure_NotDuplicate_against(row, ref_df):
    global id_col
    hash_val = utils_preprocess.get_hash_aux(row, id_col)
    r = utils_preprocess.is_duplicate(ref_df, hash_val)
    return not r


# ---------------------------------------------------- #
# Lexically sorted columns
# ---------------------------------------------------- #
def lexical_sort_cols(df, id_col):
    feature_columns = list(df.columns)
    feature_columns.remove(id_col)
    feature_columns = list(sorted(feature_columns))
    ord_cols = [id_col] + feature_columns
    return df[ord_cols]


# ---------------------------------------------------- #
# Clean up training data
# ---------------------------------------------------- #

def clean_train_data():
    global DIR
    global CONFIG
    global DIR_LOC
    files = get_files(DIR, 'train')
    list_df = [pd.read_csv(_file, usecols=use_cols, low_memory=False) for _file in files]
    list_df = HSCode_cleanup(list_df, DIR_LOC, CONFIG)

    list_df_1 = apply_value_filters(list_df)
    master_df = None
    for df in list_df_1:
        if master_df is None:
            master_df = pd.DataFrame(df, copy=True)
        else:
            master_df = master_df.append(
                df,
                ignore_index=True
            )

    master_df = remove_low_frequency_values(master_df)

    return master_df


# ------------------------------------------------- #
# set up testing data
# ------------------------------------------------- #
def setup_testing_data(
        test_df,
        train_df,
        col_val2id_dict
):
    global id_col
    global save_dir

    test_df = test_df.dropna()

    # Replace with None if ids are not in train_set
    feature_cols = list(test_df.columns)
    feature_cols.remove(id_col)
    feature_cols = list(sorted(feature_cols))

    for col in feature_cols:
        valid_items = list(col_val2id_dict[col].keys())
        test_df = test_df.loc[test_df[col].isin(valid_items)]

    # First convert to to ids
    for col in feature_cols:
        val2id_dict = col_val2id_dict[col]
        test_df[col] = test_df.apply(
            replace_attr_with_id,
            axis=1,
            args=(
                col,
                val2id_dict,
            )
        )
    test_df = test_df.dropna()
    test_df = test_df.drop_duplicates()
    test_df = lexical_sort_cols(test_df, id_col)

    print(' Length of testing data', len(test_df))

    '''
    Remove duplicates w.r.t to train set:
    Paralleleize the process 
    '''

    def aux_validate(target_df, ref_df):
        tmp_df = pd.DataFrame(
            target_df,
            copy=True
        )

        tmp_df['valid'] = tmp_df.parallel_apply(
            ensure_NotDuplicate_against,
            axis=1,
            args=(ref_df,)
        )
        print(tmp_df)
        tmp_df = tmp_df.loc[(tmp_df['valid'] == True)]
        del tmp_df['valid']
        return pd.DataFrame(tmp_df, copy=True)

    ref_df = utils_preprocess.add_hash(
        train_df.copy(), id_col
    )
    new_test_df = aux_validate(test_df, ref_df)

    print(' After deduplication :: ', len(new_test_df))
    return new_test_df


def create_train_test_sets():
    global use_cols
    global DIR
    global save_dir
    global column_value_filters
    global CONFIG
    global DIR_LOC

    train_df_file = os.path.join(save_dir, 'train_data.csv')
    test_df_file = os.path.join(save_dir, 'test_data.csv')
    column_valuesId_dict_file = 'column_valuesId_dict.pkl'
    column_valuesId_dict_path = os.path.join(save_dir, column_valuesId_dict_file)

    # --- Later on - remove using the saved file ---- #
    if os.path.exists(train_df_file) and os.path.exists(test_df_file) and False:
        train_df = pd.read_csv(train_df_file)
        test_df = pd.read_csv(test_df_file)
        with open(column_valuesId_dict_path, 'rb') as fh:
            col_val2id_dict = pickle.load(fh)

        return train_df, test_df, col_val2id_dict

    train_df = clean_train_data()
    train_df, col_val2id_dict = convert_to_ids(
        train_df,
        save_dir
    )
    print('Length of train data ', len(train_df))
    train_df = lexical_sort_cols(train_df, id_col)

    '''
         test data preprocessing
    '''
    # combine test data into 1 file :
    test_files = get_files(DIR, 'test')
    list_test_df = [
        pd.read_csv(_file, low_memory=False, usecols=use_cols)
        for _file in test_files
    ]
    list_test_df = HSCode_cleanup(list_test_df, DIR_LOC, CONFIG)

    test_df = None
    for _df in list_test_df:
        if test_df is None:
            test_df = _df
        else:
            test_df = test_df.append(_df)

    print('size of  Test set ', len(test_df))
    test_df = setup_testing_data(
        test_df,
        train_df,
        col_val2id_dict
    )

    test_df.to_csv(test_df_file, index=False)
    train_df.to_csv(train_df_file, index=False)

    # -----------------------
    # Save col_val2id_dict
    # -----------------------
    with open(column_valuesId_dict_path, 'wb') as fh:
        pickle.dump(col_val2id_dict, fh, pickle.HIGHEST_PROTOCOL)

    return train_df, test_df, col_val2id_dict


# -------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3' ],
    default= 'us_import2'
)

args = parser.parse_args()
DIR = args.DIR
# -------------------------------- #

set_up_config(args.DIR)
create_train_test_sets()
