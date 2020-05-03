#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

import argparse
import yaml
import os
import sys
import pandas as pd
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pandarallel import pandarallel
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score

pandarallel.initialize(progress_bar=False)

DIR = None
config_file = 'config.yaml'
model_use_data_DIR = None
CONFIG = None
serial_mapping_df_file = None
SOURCE_DATA_DIR_1 = None
SOURCE_DATA_DIR_2 = None
label_col = 'y'
ground_truth_col = 'y_true'
fraud_flag_col = 'fraud'
anom_score_col = 'score'
anomaly_flag_col = 'anomaly'

id_col = 'PanjivaRecordID'


def set_up_config(_DIR):
    global DIR
    global CONFIG
    global config_file
    global model_use_data_DIR
    global serial_mapping_df_file
    global SOURCE_DATA_DIR_1
    global SOURCE_DATA_DIR_2

    DIR = _DIR
    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)

    SOURCE_DATA_DIR_1 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_1']
    )

    SOURCE_DATA_DIR_2 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_2']
    )

    model_use_data_DIR = CONFIG['model_use_data_DIR']
    if not os.path.exists(CONFIG['model_use_data_DIR']):
        os.mkdir(CONFIG['model_use_data_DIR'])
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)

    mapping_df_file = CONFIG['serial_mapping_df']  # Serialized_Mapping.csv'
    serial_mapping_df_file = os.path.join(model_use_data_DIR, mapping_df_file)
    create_serialization()

    return


def convert_to_SerialID(_row, cols):
    global serial_mapping_df
    row = _row.copy()
    for c in cols:
        val = row[c]
        res = list(
            serial_mapping_df.loc[
                (serial_mapping_df['Domain'] == c) &
                (serial_mapping_df['Entity_ID'] == val)]
            ['Serial_ID']
        )
        row[c] = res[0]
    return row


def create_serialization():
    global SOURCE_DATA_DIR_1
    global DIR
    global serial_mapping_df_file
    global serial_mapping_df

    domain_dims_dict = get_domain_dims_dict()
    if os.path.exists(serial_mapping_df_file):
        serial_mapping_df = pd.read_csv(
            serial_mapping_df_file, index_col=None
        )
    else:
        prev_count = 0
        res = []
        for dn, ds in domain_dims_dict.items():
            for eid in range(ds):
                r = [dn, eid, eid + prev_count]
                res.append(r)
            prev_count += ds

        serial_mapping_df = pd.DataFrame(
            data=res,
            columns=['Domain', 'Entity_ID', 'Serial_ID']
        )

        serial_mapping_df.to_csv(
            serial_mapping_df_file,
            index=False
        )
    return


def get_domain_dims_dict():
    global SOURCE_DATA_DIR_1
    global DIR
    fpath_dd = os.path.join(
        SOURCE_DATA_DIR_1, DIR, 'domain_dims.pkl'
    )
    with open(fpath_dd, 'rb') as fh:
        domain_dims_dict = pickle.load(fh)
    return domain_dims_dict


def read_target_data():
    global SOURCE_DATA_DIR_2
    global DIR
    global label_col
    global ground_truth_col
    global fraud_flag_col

    domain_dims_dict = get_domain_dims_dict()
    fpath = os.path.join(
        SOURCE_DATA_DIR_2,
        DIR,
        'scored_test_data.csv'
    )
    df = pd.read_csv(fpath, index_col=None)

    # Serialize df
    cols = list(domain_dims_dict.keys())

    # df_s = df.parallel_apply(
    #     convert_to_SerialID,
    #     axis =1,
    #     args = (cols,)
    # )

    def set_ground_truth(val):
        if val:
            return 1
        else:
            return 0

    # set ground truths
    df[ground_truth_col] = df[fraud_flag_col].parallel_apply(
        set_ground_truth
    )
    # Valid labels are 0 and 1
    df[label_col] = -1
    df_s = df.sort_values(by='score')
    return df_s


def main_process():
    global id_col
    global clf_type
    global label_col
    global anomaly_flag_col
    global score_col
    global fraud_flag_col
    df_master = read_target_data()
    percent_labelled_list = [5,10,15,20]
    for label_perc in percent_labelled_list:
        df = df_master.copy()
        print('------')
        print("percentage of data labelled ", label_perc)
        # Convert to one hot encoding
        domain_dims = get_domain_dims_dict()
        feature_cols = list(domain_dims.keys())

        for col in feature_cols:
            df = pd.get_dummies(
                df,
                columns=[col]
            )
        _count = int(len(df) * label_perc / 100)
        df_L = df.head(_count)
        labelled_ids = list(df_L[id_col])
        df_U = df.loc[~df[id_col].isin(labelled_ids)]

        y_train = np.array(list(df_L[ground_truth_col]))
        y_test_true = list(df_U[ground_truth_col])
        test_scores = list(df_U[anom_score_col])
        test_ids = list(df_U[id_col])

        rmv_cols = [id_col, fraud_flag_col, label_col, anomaly_flag_col, ground_truth_col, anom_score_col]
        clf = None
        if clf_type == 'RF':
            clf = RandomForestClassifier()
        elif clf_type == 'SVM':
            clf = SVC()
        for rc in rmv_cols:
            del df_L[rc]
            del df_U[rc]

        X_train = df_L.values
        X_test = df_U.values
        # Train model
        clf.fit(
            X_train, y_train
        )
        y_test_pred = clf.predict(X_test)

        res_df =  pd.DataFrame(
            np.stack([test_ids, y_test_true, y_test_pred, test_scores],axis=1),
            columns = [id_col, 'y_true', 'y_pred', 'score']
        )
        res_df = res_df.sort_values(by=['score'])
        for chkpt_nxt in [10,20,30,40]:
            _count = int (len(df_master) * chkpt_nxt/100)
            _tmp = res_df.head(_count)
            y_t = list(_tmp['y_true'])
            y_p = list(_tmp['y_pred'])

            precision = precision_score(y_t, y_p)
            recall = recall_score( y_t, y_p )
            bal_acc = balanced_accuracy_score(y_t,y_p)
            f1 = 2 * precision * recall / (precision + recall)
            msg = " Next {} %; precision {} , recall {}, f1 {}, balanced_accuracy {}".format(
                chkpt_nxt, round(precision,3), round(recall,3), round(f1,3),round(bal_acc,3)
            )
            print(msg)





parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)
parser.add_argument(
    '--clf_type', choices=['RF', 'SVM'],
    default='RF'
)

args = parser.parse_args()
clf_type = args.clf_type
set_up_config(args.DIR)
main_process()