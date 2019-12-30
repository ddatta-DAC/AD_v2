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


# # Let us create APE style test & training sets
# ## us import :  Train on 2015(01-07) Test(08-09)


def get_regex(_type):
    global DIR
    if DIR == 'us_import':
        if _type == 'train':
            return '*0[1-4]**2015*.csv'
        if _type == 'test':
            return '*0[5-6]*2015*.csv'
    if DIR == 'us_import2':
        if _type == 'train':
            return '*0[1-6]**2015*.csv'
        if _type == 'test':
            return '*0[7-9]**2015*.csv'
    if DIR == 'us_import3':
        if _type == 'train':
            return '*0[1-4]**2015*.csv'
        if _type == 'test':
            return '*0[5-6]**2015*.csv'
    if DIR == 'china_import':
        if _type == 'train':
            return '*0[1-7]*2015*.csv'
        if _type == 'test':
            return '*0[1-7]*2016*.csv'

    if DIR == 'china_import2':
        if _type == 'train':
            return '*0[1-9]*2015*.csv'
        if _type == 'test':
            return '*0[1-6]*2016*.csv'

    if DIR == 'peru_export':
        if _type == 'train':
            return '*201[5-6]*.csv'
        if _type == 'test':
            return '*201[7,8]*.csv'

    if DIR == 'peru_export2':
        if _type == 'train':
            return '*201[5-6]*.csv'
        if _type == 'test':
            return '*201[7,8]*.csv'

    if DIR == 'china_export':
        if _type == 'train':
            return '*0[1-4]*2015*.csv'
        if _type == 'test':
            return '*0[5-6]*2015*.csv'

    return '*.csv'


def get_files(_type='all'):
    global DIR
    data_dir = os.path.join(
        './../../wwf_data_v1',
        DIR
    )

    regex = get_regex(_type)
    files = sorted(
        glob.glob(
            os.path.join(data_dir, regex)
        )
    )
    print(files)
    return files


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
    domain_dims_dict = {}
    col_val2id_dict = {}

    for col in sorted(feature_columns):
        vals = list(set(df[col]))

        # { 0 : item1 , 1 :item2, ... }
        id2val_dict = {
            e[0]: e[1]
            for e in enumerate(vals, 0)
        }

        # { item1 : 0, item2 : 1, ... }
        val2id_dict = {
            v: k for k, v in id2val_dict.items()
        }
        col_val2id_dict[col] = val2id_dict

        # replace
        df[col] = df.apply(
            replace_attr_with_id,
            axis=1,
            args=(
                col,
                val2id_dict,
            )
        )
        domain_dims_dict[col] = len(id2val_dict)

    domain_dims = []
    domain_dims_res = {}

    print(list(df.columns))

    for col in list(df.columns):
        if col in domain_dims_dict.keys():
            print(col)
            domain_dims_res[col] = domain_dims_dict[col]
            domain_dims.append(domain_dims_dict[col])

    domain_dims = np.array(domain_dims)
    print(domain_dims_res)

    file = 'domain_dims.pkl'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    f_path = os.path.join(save_dir, file)

    with open(f_path, 'wb') as fh:
        pickle.dump(
            domain_dims_res,
            fh,
            pickle.HIGHEST_PROTOCOL
        )
    return df, col_val2id_dict


def collate(file_list):
    global id_col
    global use_cols
    print(use_cols)

    _master_df = None
    for file in file_list:
        _df = pd.read_csv(
            file,
            low_memory=False,
            usecols=use_cols
        )
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
    print(all_cols)
    _master_df = _master_df[all_cols]
    return _master_df


def remove_low_frequency_values(_df):
    global id_col
    global freq_bound
    from collections import Counter

    freq_column_value_filters = {}

    feature_cols = list(_df.columns)
    feature_cols.remove(id_col)

    for c in feature_cols:
        values = list(_df[c])
        freq_column_value_filters[c] = []

        obj_counter = Counter(values)

        for _item, _count in obj_counter.items():
            if _count < freq_bound:
                freq_column_value_filters[c].append(_item)

    for c, _items in freq_column_value_filters.items():
        print(c, len(_items))
    print(len(_df))
    for col, val in freq_column_value_filters.items():
        _df = _df.loc[
            (~_df[col].isin(val))
        ]

    return _df


def validate(row, ref_df):
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


'''
returns c random items as a dict
column_name : item_id
'''


def get_c_vals(anomaly_cols, col_val2id_dict):
    res_dict = {}
    for col in anomaly_cols:
        res_dict[col] = random.sample(list(col_val2id_dict[col].values()), 1)[0]
    return res_dict


def create_anomalies(test_df, train_df, col_val2id_dict, c=3):
    global id_col
    feature_cols = list(test_df.columns)
    feature_cols.remove(id_col)
    feature_cols_id = {e[0]: e[1] for e in enumerate(feature_cols)}
    ref_df = pd.DataFrame(train_df, copy=True)
    ref_df = ref_df.append(
        test_df,
        ignore_index=True
    )
    new_df = pd.DataFrame(columns=list(test_df.columns))

    for i, row in test_df.iterrows():
        while True:
            _anomaly_cols = [feature_cols_id[_]
                             for _ in random.sample(
                    list(feature_cols_id.keys()),
                    k=c
                )
                             ]
            c_vals = get_c_vals(_anomaly_cols, col_val2id_dict)
            row_copy = pd.Series(row, copy=True)
            for _col, _item_id in c_vals.items():
                row_copy[_col] = _item_id
            if validate(row_copy, ref_df):
                row_copy[id_col] = int(str(row_copy[id_col]) + '01' + str(c))
                new_df = new_df.append(row_copy, ignore_index=True)
                break

    # sample c cols
    new_df = new_df.drop_duplicates(subset=feature_cols)
    print(' Length of anomalies_df ', new_df)
    return (c, new_df)


def setup_testing_data(test_df, train_df, col_val2id_dict):
    global id_col
    # Replace with None if ids are not in train_set
    print('----')
    feature_cols = list(test_df.columns)
    feature_cols.remove(id_col)

    for col in feature_cols:
        valid_items = list(col_val2id_dict[col].keys())
        test_df = test_df.loc[test_df[col].isin(valid_items)]

    print(' Length of testing data', len(test_df))

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

    '''
    Remove duplicates :
    '''

    print(' Length of test df :: ', len(test_df))
    new_test_df = pd.DataFrame(columns=list(test_df.columns))

    for i, row in test_df.iterrows():
        if validate(row, train_df):
            new_test_df = new_test_df.append(row, ignore_index=True)

    print(' After deduplication :: ', len(new_test_df))

    results = Parallel(n_jobs=3)(
        delayed(create_anomalies)(
            new_test_df, train_df, col_val2id_dict, c=_i)
        for _i in range(1, 4)
    )
    anomalies_df_c1 = None
    anomalies_df_c2 = None
    anomalies_df_c3 = None

    for _result in results:
        if _result[0] == 1:
            anomalies_df_c1 = _result[1]
        elif _result[0] == 2:
            anomalies_df_c2 = _result[1]
        elif _result[0] == 3:
            anomalies_df_c3 = _result[1]

    return new_test_df, anomalies_df_c1, anomalies_df_c2, anomalies_df_c3


def create_train_test_sets():
    global use_cols
    global DIR
    global save_dir
    global column_value_filters
    train_files = get_files('train')
    test_files = get_files('test')
    global save_dir
    # combine train_data :
    train_master_df = collate(train_files)
    test_master_df = collate(test_files)

    print(' Train initial ', len(train_master_df))
    print(' Test initial ', len(test_master_df))

    '''
    test data preprocessing
    '''
    print(len(train_master_df))

    '''
    Remove values that are garbage
    '''
    if type(column_value_filters) != bool:
        for col, val in column_value_filters.items():
            train_master_df = train_master_df.loc[
                (~train_master_df[col].isin(val))
            ]

    print(' Length of training data ', len(train_master_df))

    train_master_df = remove_low_frequency_values(
        train_master_df
    )

    train_master_df_1, col_val2id_dict = convert_to_ids(
        train_master_df,
        save_dir
    )

    new_test_df, anomalies_df_c1, anomalies_df_c2, anomalies_df_c3 = setup_testing_data(
        test_master_df,
        train_master_df_1,
        col_val2id_dict
    )

    new_test_df.to_csv(os.path.join(save_dir, 'test_data.csv'), index=False)
    train_master_df_1.to_csv(os.path.join(save_dir, 'train_data.csv'), index=False)
    anomalies_df_c1.to_csv(os.path.join(save_dir, 'anomalies_c1_data.csv'), index=False)
    anomalies_df_c2.to_csv(os.path.join(save_dir, 'anomalies_c2_data.csv'), index=False)
    anomalies_df_c3.to_csv(os.path.join(save_dir, 'anomalies_c3_data.csv'), index=False)

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


'''
Create numpy arrays 
Same test & anomaly data will be shared by the models
Store in .pkl files
'''


def create_ape_model_data():
    global DIR
    global term_2_col
    global term_4_col
    global save_dir
    global id_col
    global ns_id_col
    global num_neg_samples_ape

    train_pos_data_file = os.path.join(save_dir, 'train_data.csv')
    train_neg_data_file = os.path.join(save_dir, 'negative_samples_ape_1.csv')
    test_data_file = os.path.join(save_dir, 'test_data.csv')
    anomalies_c1_data_file = os.path.join(save_dir, 'anomalies_c1_data.csv')
    anomalies_c2_data_file = os.path.join(save_dir, 'anomalies_c2_data.csv')
    anomalies_c3_data_file = os.path.join(save_dir, 'anomalies_c3_data.csv')

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

    anomalies_c1_df = pd.read_csv(
        anomalies_c1_data_file,
        index_col=None
    )

    anomalies_c2_df = pd.read_csv(
        anomalies_c2_data_file,
        index_col=None
    )

    anomalies_c3_df = pd.read_csv(
        anomalies_c3_data_file,
        index_col=None
    )

    feature_cols = list(train_pos_df.columns)
    feature_cols.remove(id_col)

    # Anomalies generated have fake panjiva id
    test_anomaly_c1_idList = list(anomalies_c1_df[id_col])
    test_anomaly_c2_idList = list(anomalies_c2_df[id_col])
    test_anomaly_c3_idList = list(anomalies_c3_df[id_col])

    test_normal_idList = list(test_df[id_col])

    try:
        del test_df[id_col]
        del anomalies_c1_df[id_col]
        del anomalies_c2_df[id_col]
        del anomalies_c3_df[id_col]
    except:
        pass

    matrix_test = test_df.values
    matrix_anomaly_c1 = anomalies_c1_df.values
    matrix_anomaly_c2 = anomalies_c2_df.values
    matrix_anomaly_c3 = anomalies_c3_df.values

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
    f_path = os.path.join(
        save_dir,
        'matrix_train_positive.pkl'
    )

    with open(f_path, 'wb') as fh:
        pickle.dump(
            matrix_pos,
            fh,
            pickle.HIGHEST_PROTOCOL
        )
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

    f_path = os.path.join(save_dir, 'matrix_test_positive.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            matrix_test,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    f_path = os.path.join(save_dir, 'matrix_test_anomalies_c1.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            matrix_anomaly_c1,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    f_path = os.path.join(save_dir, 'matrix_test_anomalies_c2.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            matrix_anomaly_c2,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    f_path = os.path.join(save_dir, 'matrix_test_anomalies_c3.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            matrix_anomaly_c3,
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    f_path = os.path.join(save_dir, 'test_idList_c1.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            [test_anomaly_c1_idList, test_normal_idList],
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    f_path = os.path.join(save_dir, 'test_idList_c2.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            [test_anomaly_c2_idList, test_normal_idList],
            fh,
            pickle.HIGHEST_PROTOCOL
        )

    f_path = os.path.join(save_dir, 'test_idList_c3.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            [test_anomaly_c3_idList, test_normal_idList],
            fh,
            pickle.HIGHEST_PROTOCOL
        )


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

        if validate(new_row, ref_df):
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

    num_chunks = 40
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
    feature_cols_id = {
        e[0]: e[1]
        for e in enumerate(feature_cols)
    }

    # get the domain dimensions
    with open(
            os.path.join(save_dir, 'domain_dims.pkl'), 'rb'
    ) as fh:
        domain_dims = pickle.load(fh)

        # Store what are valid values for each columns
    column_valid_values = {}
    for _fc_name in feature_cols:
        column_valid_values[_fc_name] = list(set(list(ref_df[_fc_name])))

    chunk_len = int(len(train_df) / (num_chunks - 1))

    list_df_chunks = np.split(
        train_df.head(
            chunk_len * (num_chunks - 1)
        ), num_chunks - 1
    )

    end_len = len(train_df) - chunk_len * (num_chunks - 1)
    list_df_chunks.append(train_df.tail(end_len))
    for _l in range(len(list_df_chunks)):
        print(len(list_df_chunks[_l]), _l)

    results = []

    results = Parallel(n_jobs=10)(
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
    for _f in results:
        _df = pd.read_csv(_f, index_col=None)

        if new_df is None:
            new_df = _df
        else:
            new_df = new_df.append(_df, ignore_index=True)
        print(' >> ', len(new_df))

    new_df.to_csv(os.path.join(save_dir, 'negative_samples_v1.csv'), index=False)
    return new_df


'''
Save the pickle files to be used by model 
Specifically the negative samples and train data 
Uses same source but the order may vary
'''


def create_model_data_v1():
    global DIR
    global term_2_col
    global term_4_col
    global save_dir
    global id_col
    global ns_id_col
    global num_neg_samples_v1

    train_pos_data_file = os.path.join(save_dir, 'train_data.csv')
    train_neg_data_file = os.path.join(save_dir, 'negative_samples_v1.csv')

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
    f_path = os.path.join(
        save_dir,
        'matrix_train_positive_v1.pkl'
    )

    with open(f_path, 'wb') as fh:
        pickle.dump(
            matrix_pos,
            fh,
            pickle.HIGHEST_PROTOCOL
        )
    f_path = os.path.join(save_dir, 'negative_samples_v1.pkl')
    with open(f_path, 'wb') as fh:
        pickle.dump(
            matrix_neg,
            fh,
            pickle.HIGHEST_PROTOCOL
        )


def main():
    set_up_config()
    create_train_test_sets()
    create_negative_samples_ape()
    create_ape_model_data()
    create_negative_samples_v1()
    create_model_data_v1()
    return

main()
