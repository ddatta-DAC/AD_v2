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
from collections import Counter
sys.path.append('.')
sys.path.append('./..')
try:
    import clean_up_test_data
except:
    from . import clean_up_test_data
try:
    import create_anomalies
except:
    from . import create_anomalies

# ===================

CONFIG_FILE = 'config_preprocessor_v02.yaml'
id_col = 'PanjivaRecordID'
ns_id_col = 'NegSampleID'
term_2_col = 'term_2'
term_4_col = 'term_4'
num_neg_samples_ape = None
use_cols = None
freq_bound = None
column_value_filters = None
num_neg_samples = None
save_dir = None
cleaned_csv_subdir = None


# ====================

def get_regex(_type):
    global DIR

    if DIR == 'us_import':
        if _type == 'train':
            return '*0[1-6]*2016*.csv'
        if _type == 'test':
            return '*0[7-9]*2016*.csv'

    if DIR == 'china_import':
        if _type == 'train':
            return '*0[1-6]*2016*.csv'
        if _type == 'test':
            return '*0[7-9]*2016*.csv'

    if DIR == 'china_export':
        if _type == 'train':
            return '*0[1-4]*2016*.csv'
        if _type == 'test':
            return '*0[5-6]*2016*.csv'

    return '*.csv'


def get_files(DIR, _type='all'):
    data_dir = os.path.join(
        './../../Data_Raw',
        DIR
    )

    regex = get_regex(_type)
    files = sorted(
        glob.glob(
            os.path.join(data_dir, regex)
        )
    )
    print('DIR ::', DIR, ' Type ::', _type, 'Files count::', len(files))
    return files


def set_up_config():
    global CONFIG_FILE
    global use_cols
    global freq_bound
    global num_neg_samples_ape
    global DIR
    global save_dir
    global column_value_filters
    global num_neg_samples
    global cleaned_csv_subdir

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    DIR = CONFIG['DIR']
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

    return CONFIG


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

    feature_columns = list(df.columns)
    feature_columns.remove(id_col)

    dict_DomainDims = {}
    col_val2id_dict = {}

    for col in sorted(feature_columns):
        vals = list(set(df[col]))

        # ----
        #
        #   0 : item1 ,
        #   1 : item2 ,
        #   ...
        # ----
        id2val_dict = {
            e[0]: e[1]
            for e in enumerate(vals, 0)
        }

        # ----
        #
        #   item1 : 0 ,
        #   item2 : 0 ,
        #   ...
        # ----
        val2id_dict = {
            v: k for k, v in id2val_dict.items()
        }
        col_val2id_dict[col] = val2id_dict

        # Replace
        df[col] = df.apply(
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
    return df, col_val2id_dict


'''
Join the csv files into 1 single Dataframe
Removes missing values
Input : file_path_list
'''


def collate(file_list):
    global id_col
    global use_cols

    _master_df = None
    for file in file_list:
        _df = pd.read_csv(
            file,
            low_memory=False,
            usecols=use_cols
        )

        # Drop missing values
        _df = _df.dropna()
        if _master_df is None:
            _master_df = pd.DataFrame(_df)
        else:
            _master_df = _master_df.append(
                _df,
                ignore_index=True
            )

    feature_cols = list(_master_df.columns)
    feature_cols.remove(id_col)
    feature_cols = list(sorted(feature_cols))

    all_cols = [id_col]
    all_cols.extend(feature_cols)
    print(' Columns in the dataframe : ', all_cols)
    _master_df = _master_df[all_cols]
    return _master_df


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
        df = df.loc[
            (~df[col].isin(val))
        ]

    return df


def HSCode_cleanup(list_df, DIR, config):
    hscode_col = 'HSCode'

    # ----- #
    # Expert curated HS codes
    hs_code_filter_file = os.path.join(config['hscode_filter_file_loc'], DIR + config['hscode_filter_file_pattern'])
    tmp = pd.read_csv(hs_code_filter_file, index_col=None, header=None)
    target_codes = list(tmp[0])

    def hsc_proc(_code):
        return str(_code)[:4]

    target_codes = list(sorted([hsc_proc(_) for _ in target_codes]))

    def filter_by_ExpertHSCodeList(_code, target_codes):
        if _code[:2] in target_codes or _code[:4] in target_codes:
            return _code
        return None

    # ------ #
    # Correct the formats of HSCodes :
    # eg. in china_export add in the preceeding 0
    # ------ #
    def add_preceeding_zero(_code):
        _code = _code.strip()
        if len(_code) > 6:
            _code = _code[:6]
        elif len(_code) == 5:
            _code = '0' + _code
        return _code

    list_processed_df = []
    for df in list_df:
        df = df.dropna()
        df[hscode_col] = df[hscode_col].astype(str)
        if DIR == 'china_export':
            df[hscode_col] = df[hscode_col].apply(add_preceeding_zero)

        df[hscode_col] = df[hscode_col].apply(
            filter_by_ExpertHSCodeList,
            args=(target_codes,)
        )
        df = df.dropna()
        list_processed_df.append(df)
    # --------- #

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
                df = df.loc[(~df[col].isin(val))]
            list_processed_df.append(df)
        return list_processed_df
    return list_df


def ensure_NotDuplicate_against(row, ref_df):
    global id_col
    query_str = []

    for _c, _i in row.to_dict().items():
        if _c == id_col:
            continue
        query_str.append(' ' + _c + ' == ' + str(_i))
    query_str = ' & '.join(query_str)
    res_query = ref_df.query(query_str)

    if len(res_query) > 0:
        return False
    return True


def clean_train_data():
    global DIR
    global CONFIG

    files = get_files(DIR, 'train')
    list_df = [pd.read_csv(_file, usecols=use_cols, low_memory=False) for _file in files]
    list_df = HSCode_cleanup(list_df, DIR, CONFIG)

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


def setup_testing_data(
        test_df,
        train_df,
        col_val2id_dict
):
    global id_col
    global save_dir

    # Replace with None if ids are not in train_set
    feature_cols = list(test_df.columns)
    feature_cols.remove(id_col)
    test_df = test_df.dropna()

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
    print(' Length of testing data', len(test_df))

    '''
    Remove duplicates w.r.t to train set:
    Paralleleize the process 
    '''

    def aux_validate(target_df, train_df):
        tmp_df = pd.DataFrame(
            target_df,
            copy = True
        )

        tmp_df['valid'] = tmp_df.apply(
            ensure_NotDuplicate_against,
            axis=1,
            args=(train_df,)
        )

        tmp_df['valid'] = tmp_df.loc[ (tmp_df['valid']==True) ]
        del tmp_df['valid']
        return pd.DataFrame(tmp_df,copy=True)


    num_chunks = 40
    chunk_len = int(len(test_df) // num_chunks)

    list_df_chunks = np.split(
        test_df.head(chunk_len * (num_chunks - 1)), num_chunks - 1
    )

    end_len = len(test_df) - chunk_len * (num_chunks - 1)
    list_df_chunks.append(test_df.tail(end_len))

    print(' Deduplication of test set w.r.t. train :: Length of chunks ',
          [len(_) for _ in list_df_chunks])

    list_dedup_df = Parallel(n_jobs=num_chunks)(
        delayed(aux_validate)(target_df, train_df)
        for target_df in list_df_chunks
    )

    new_test_df = None
    for _df in list_dedup_df:
        if new_test_df is None:
            new_test_df = _df
        else:
            new_test_df = new_test_df.append(_df, ignore_index=True)

    print(' After deduplication :: ', len(new_test_df))
    return new_test_df


def create_train_test_sets():
    global use_cols
    global DIR
    global save_dir
    global column_value_filters
    global CONFIG

    train_df_file = os.path.join(save_dir, 'train_data.csv')
    test_df_file = os.path.join(save_dir, 'test_data.csv')
    column_valuesId_dict_file = 'column_valuesId_dict.pkl'
    column_valuesId_dict_path = os.path.join(save_dir, column_valuesId_dict_file)
    # --- Later on - remove using the saved file ---- #
    if os.path.exists(train_df_file) and os.path.exists(test_df_file):
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
    train_df.to_csv(train_df_file, index=False)

    '''
         test data preprocessing
    '''
    # combine test data into 1 file :
    test_files = get_files(DIR, 'test')
    list_test_df = [
        pd.read_csv(_file, low_memory=False, usecols=use_cols)
        for _file in test_files
    ]
    list_test_df = HSCode_cleanup(list_test_df, DIR, CONFIG)

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

    # Save col_val2id_dict

    with open(column_valuesId_dict_path,'wb') as fh:
        pickle.dump(col_val2id_dict, fh, pickle.HIGHEST_PROTOCOL)

    return train_df, test_df, col_val2id_dict


# -------------------------------#

def clean_test_data_level2( ):
    global save_dir
    global CONFIG
    global id_col

    train_df_file = os.path.join(save_dir, 'train_data.csv')
    test_df_file = os.path.join(save_dir, 'test_data.csv')
    train_df = pd.read_csv(train_df_file)
    test_df = pd.read_csv(test_df_file)

    test_df = clean_up_test_data.remove_order1_spurious_coocc(
        train_df,
        test_df,
        id_col
    )

    test_df_file = os.path.join(save_dir, CONFIG['test_data_file_v1'])
    test_df.to_csv(test_df_file, index=False)
    return


CONFIG = set_up_config()
# create_train_test_sets()
# clean_test_data_level2()

train_df = pd.read_csv(os.path.join(save_dir, CONFIG['train_data_file']))
test_df = pd.read_csv(os.path.join(save_dir, CONFIG['test_data_file_v1']))

create_anomalies.generate_type1_anomalies(
        test_df,
        train_df,
        save_dir,
        id_col,
        num_jobs=40,
        anom_perc=10
)


create_anomalies.generate_type2_anomalies(
        test_df,
        train_df,
        save_dir,
        id_col,
        num_jobs=40,
        anom_perc=10
)


create_anomalies.generate_type3_anomalies(
        test_df,
        train_df,
        save_dir,
        id_col,
        num_jobs=40,
        anom_perc=10
)


