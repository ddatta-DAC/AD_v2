from src.Classifiers.wide_n_deep_model import clf_wide_n_deep as clf_WIDE_N_DEEP
from src.Classifiers.deepFM import clf_deepFM as clf_DEEP_FM

from src.Classifiers import wide_n_deep_model
from src.Classifiers import deepFM

import pandas as pd
import numpy as np
import os
import pickle


def convert_to_serial_IDs(
        df,
        serial_mapping_df,
        keep_entity_ids=True,
        domain_list=None
):
    reference_dict = {}
    for d in set(serial_mapping_df['Domain']):
        reference_dict[d] = {}
        _tmp = serial_mapping_df.loc[(serial_mapping_df['Domain'] == d)]
        k = _tmp['Entity_ID']
        v = _tmp['Serial_ID']
        reference_dict[d] = {_k: _v for _k, _v in zip(k, v)}

    # Inplace conversion
    def aux_conv_toSerialID(_row):
        row = _row.copy()
        for fc in domain_list:
            col_name = fc
            if keep_entity_ids:
                col_name = '_' + fc
            row[col_name] = reference_dict[fc][row[fc]]
        return row

    df = df.parallel_apply(aux_conv_toSerialID, axis=1)
    if keep_entity_ids:
        new_features = ['_' + d for d in domain_list]
        return df, new_features
    else:
        return df


# Get the normal "training" data
def get_base_train_data(
        src_dir,
        fraud_col='fraud',
        anomaly_col='anomaly',
        label_col='y',
        is_labelled_col='labelled',
        true_label_col='y_true',
):
    df = pd.read_csv(
        os.path.join(src_dir, 'train_data.csv'), index_col=None
    )
    df[fraud_col] = False
    df[anomaly_col] = False
    df[true_label_col] = 0
    df[is_labelled_col] = True
    df[label_col] = 0.
    df = df.sample(frac=0.05)
    return df


def get_base_train_data_sample(
        df,
        data_size=2000
):
    return df.sample(data_size)


def PreProcessData(
        data_df_list,
        clf_type,
        domain_dims,
        serial_mapping_df,
        model_use_data_DIR,
        id_col='PanjivaRecordID'
):
    if clf_type is None:
        clf_type = 'MLP'

    df_fpath = os.path.join(model_use_data_DIR, 'preprocessed_data_' + clf_type + '.csv')
    features_f_file = os.path.join(model_use_data_DIR, 'features_F_' + clf_type + '.dat')
    features_g_file = os.path.join(model_use_data_DIR, 'features_G_' + clf_type + '.dat')
    id_pool_file = os.path.join(model_use_data_DIR, '__id_pool_' + clf_type + '.dat')
    print('Files: ', df_fpath, features_f_file, features_g_file, id_pool_file)

    if os.path.exists(df_fpath) and os.path.exists(id_pool_file):

        # Save the data in a file
        converted_df = pd.read_csv(df_fpath, index_col=None)
        with open(id_pool_file, 'rb') as fh:
            id_pool = pickle.load(fh)
        # Save the list of features
        with open(features_f_file, 'rb') as fh:
            features_F = pickle.load(fh)
        with open(features_g_file, 'rb') as fh:
            features_G = pickle.load(fh)

        df_list = []
        for id_list in id_pool:
            df_list.append(converted_df.loc[converted_df[id_col].isin(id_list)].copy())
        df_target = df_list[0]
        normal_data_samples_df = df_list[1]
        return df_target, normal_data_samples_df, features_F, features_G

    # set up features of Classifier and Agreement model
    features_F = None
    features_G = None
    domain_list = list(domain_dims.keys())

    df_data = None
    id_pool = []
    converted_df = None
    for _df in data_df_list:
        ids = list(_df[id_col])
        id_pool.append(ids)
        if df_data is None:
            df_data = _df
        else:
            df_data = df_data.append(_df, ignore_index=True)

    if clf_type == 'wide_n_deep':
        cross_pair_list = [
            ['ShipmentOrigin', 'PortOfLading'],
            ['ShipmentDestination', 'PortOfUnlading'],
            ['ShipmentOrigin', 'HSCode'],
            ['ShipmentDestination', 'HSCode'],
            ['PortOfUnlading', 'PortOfLading'],
        ]

        converted_df = wide_n_deep_model.wide_N_deep_data_preprocess(
            df=df_data,
            domain_dims=domain_dims,
            pairs=cross_pair_list,
            remove_orig_nonserial=False,
            id_col=id_col
        )

        # The added domains are the ones to be fed to the linear layer
        features_F = [_ for _ in list(converted_df.columns) if _ not in list(df_data.columns)]

        # ----
        converted_df, features_1 = convert_to_serial_IDs(
            converted_df,
            serial_mapping_df,
            keep_entity_ids=True,
            domain_list=domain_list
        )
        # Serilaized ids fetch the embeddings
        features_G = features_1
        import re
        pattern_1 = '^[A-Z]([a-z]|[A-Z])+_[0-9]+$'
        pattern_2 = '^[A-Z]([a-z]|[A-Z])+_[A-Z]([a-z]|[A-Z])+_[0-9]+$'
        pattern_3 = '_[A-Z]([a-z]|[A-Z])+$'

        features_G = []
        features_F = []
        non_FG = []
        cols = list(converted_df.columns)
        for c in cols:
            if re.search(pattern_1, c) or re.search(pattern_2, c):
                features_F.append(c)
        for c in cols:
            if re.search(pattern_3, c):
                features_G.append(c)

        for _ in cols:
            if _ not in features_G and _ not in features_F: non_FG.append(_)
        ordered_cols = features_F + features_G + non_FG
        converted_df = converted_df[ordered_cols]
        # features_F = features_F + features_G

    elif clf_type == 'deepFM':
        if not os.path.exists(df_fpath):
            converted_df = deepFM.deepFM_data_preprocess(
                df=df_data,
                domain_dims=domain_dims,
                remove_orig_nonserial=False
            )
        else:
            converted_df = pd.read_csv(df_fpath,index_col=None)
        # Wide columns are of the format <Domain> + '_' +[<domain>_] + value


        # The added domains are the one sto be fed to the linear layer
        features_F = [_ for _ in list(converted_df.columns) if _ not in list(df_data.columns)]
        converted_df, features_1 = convert_to_serial_IDs(
            converted_df,
            serial_mapping_df,
            keep_entity_ids=True,
            domain_list=domain_list
        )

        # Serilaized ids fetch the embeddings
        import re
        pattern_1 = '^[A-Z]([a-z]|[A-Z])+_[0-9]+$'
        pattern_3 = '_[A-Z]([a-z]|[A-Z])+$'

        features_G = []
        features_F = []
        non_FG = []
        cols = list(converted_df.columns)
        for c in cols:
            if re.search(pattern_1, c) :
                features_F.append(c)
        for c in cols:
            if re.search(pattern_3, c):
                features_G.append(c)

        for _ in cols:
            if _ not in features_G and _ not in features_F: non_FG.append(_)
        ordered_cols = features_F + features_G + non_FG
        converted_df = converted_df[ordered_cols]

    elif clf_type == 'MLP':
        converted_df, features_1 = convert_to_serial_IDs(
            df_data,
            serial_mapping_df,
            keep_entity_ids=True,
            domain_list=domain_list
        )
        # Serilaized ids fetch the embeddings
        features_F = features_1
        features_G = features_1

        # Save the  file
    converted_df.to_csv(df_fpath, index=False)
    with open(id_pool_file, 'wb') as fh:
        pickle.dump(id_pool, fh, pickle.HIGHEST_PROTOCOL)

    # Save the  feature_list
    with open(features_f_file, 'wb') as fh:
        pickle.dump(features_F, fh, pickle.HIGHEST_PROTOCOL)

    with open(features_g_file, 'wb') as fh:
        pickle.dump(features_G, fh, pickle.HIGHEST_PROTOCOL)

    df_list = []
    for id_list in id_pool:
        df_list.append(
            converted_df.loc[converted_df[id_col].isin(id_list)].copy()
        )

    df_target = df_list[0]
    normal_data_samples_df = df_list[1]

    return df_target, normal_data_samples_df, features_F, features_G


def set_ground_truth_labels(
        df,
        fraud_col,
        true_label_col
):
    def aux_true_label(row):
        if row[fraud_col]:
            return 1
        else:
            return 0

    df[true_label_col] = df.parallel_apply(aux_true_label, axis=1)
    return df


# -----------------------
# Get o/p from the AD system
# -----------------------
def read_scored_data(
        DATA_SOURCE_DIR_2,
        score_col,
        label_col,
        fraud_col,
        true_label_col,
        is_labelled_col
):
    df = pd.read_csv(
        os.path.join(DATA_SOURCE_DIR_2, 'scored_test_data.csv'), index_col=None
    )
    df = df.sort_values(by=[score_col])
    df[label_col] = 0
    df = set_ground_truth_labels(df, fraud_col, true_label_col)
    df[is_labelled_col] = False
    return df


# ------------------------------------
# To be called from external file
# ------------------------------------
def get_data_plus_features(
        DATA_SOURCE_DIR_1,
        DATA_SOURCE_DIR_2,
        model_use_data_DIR,
        clf_type,
        domain_dims,
        serial_mapping_df,
        score_col,
        is_labelled_col,
        label_col,
        true_label_col,
        fraud_col,
        anomaly_col
):
    base_train_df = get_base_train_data(
        DATA_SOURCE_DIR_1,
        is_labelled_col,
        label_col,
        true_label_col,
        fraud_col,
        anomaly_col
    )

    df_target = read_scored_data(
        DATA_SOURCE_DIR_2,
        score_col,
        label_col,
        fraud_col,
        true_label_col,
        is_labelled_col
    )
    df_target = df_target
    df_target, normal_data_samples_df, features_F, features_G = PreProcessData(
        [df_target, base_train_df],
        clf_type=clf_type,
        domain_dims=domain_dims,
        serial_mapping_df=serial_mapping_df,
        model_use_data_DIR=model_use_data_DIR
    )

    return df_target, normal_data_samples_df, features_F, features_G
