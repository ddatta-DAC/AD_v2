#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

# !/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '')

# In[3]:


import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import pickle
import argparse
sys.path.append('./..')
sys.path.append('./../..')
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

pandarallel.initialize()
label_col = 'y'
fraud_flag_col = 'fraud'
anomaly_flag_col = 'anomaly'
try:
    from .src.data_preprocessor import utils_preprocess as utils
except:
    import src.data_preprocessor.utils_preprocess as utils

# In[4]:


config_file = 'config.yaml'
model_use_data_DIR = None
CONFIG = None
serial_mapping_df_file = None
serial_mapping_df = None
SOURCE_DATA_DIR_1 = None
SOURCE_DATA_DIR_2 = None
label_col = 'y'
ground_truth_col = 'y_true'
fraud_flag_col = 'fraud'
anom_score_col = 'score'
anomaly_flag_col = 'anomaly'
id_col = 'PanjivaRecordID'
DATA_DIR = None



def set_up_config(_DIR):
    global DIR
    global CONFIG
    global config_file
    global model_use_data_DIR
    global serial_mapping_df_file
    global SOURCE_DATA_DIR_1
    global SOURCE_DATA_DIR_2
    global serial_mapping_df
    global serial_mapping_file
    global DATA_DIR
    CONFIG_FILE = 'config_preprocessor_v02.yaml'
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)


    DIR = _DIR
    SOURCE_DATA_DIR_1 = './../../generated_data_v3'
    DATA_DIR = SOURCE_DATA_DIR_1

    model_use_data_DIR = 'model_use_data_DIR'
    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    return

# ====================================
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import4', 'us_import5', 'us_import6'],
    default=None
)

args = parser.parse_args()
DIR = args.DIR
set_up_config(DIR)

# =====================================



def get_data():
    global id_col
    global DATA_DIR
    global DIR
    f_name_train = 'train_data.csv'
    f_name_test = 'test_data.csv'

    df_train = pd.read_csv(os.path.join(DATA_DIR, DIR, f_name_train), index_col=None)
    df_test = pd.read_csv(os.path.join(DATA_DIR, DIR, f_name_test), index_col=None)

    with open(os.path.join(DATA_DIR, DIR, 'domain_dims.pkl'), 'rb') as fh:
        domain_dims = pickle.load(fh)

    # -------------------
    # Sort the columns
    # -------------------
    cols = list(df_train.columns)
    cols.remove((id_col))
    cols = [id_col] + list(sorted(cols))

    df_train = df_train[cols]
    df_test = df_test[cols]
    return df_train, df_test, domain_dims




def generate_by_criteria(
        row,
        criteria,
        _fixed,
        _perturb,
        anom_prob=0.5,
        id_col='PanjivaRecordID'

):
    global co_occurrence_dict
    global DATA_DIR

    new_row = row.copy()

    if np.random.random(1)[0] <= anom_prob:
        print('pertubing ....')
        size = np.random.randint(1, 3)
        p_d = np.random.choice(_perturb, size=size, replace=False)
        new_row = row.copy()
        for _dom in p_d:
            # select reference_domain from _fixed
            _f_d = np.random.choice(_fixed, size=1, replace=False)[0]
            _f_entity = row[_f_d]

            # select entity in _dom such that it does not co-occur with
            _pair = sorted([_dom, _f_d])
            key = '_+_'.join(_pair)

            _matrix = co_occurrence_dict[key]
            if _pair[0] == _f_d:
                vec = _matrix[_f_entity, :]
            else:
                vec = _matrix[:, _f_entity]
            # Select e such that vec[e] == 0
            pool = list(np.where(vec == 0)[0])
            e = np.random.choice(pool, size=1)[0]
            new_row[_dom] = e
            # check for duplicates

        suffix = '010' + str(criteria)
        new_row[id_col] = utils.aux_modify_id(new_row[id_col], suffix)
        print(criteria, ' :: generated record')

    return new_row



df_train, df_test, domain_dims = get_data()
co_occurrence_dict = utils.get_coOccMatrix_dict(df_train, id_col='PanjivaRecordID')

feature_cols = list(domain_dims.keys())
feature_cols = list(sorted(feature_cols))


# Some Origins
a1 = df_train.groupby(['ShipmentOrigin']).size().reset_index(name='count')
lb = np.percentile(list(a1['count']), 25)
ub = np.percentile(list(a1['count']), 90)
a2 = a1.loc[(a1['count'] >= lb) & (a1['count'] <= ub)]
a3 = a2.sample(frac=0.4)
a4 = a3.reset_index(drop=True)

suspect_SO = list(a4['ShipmentOrigin'])
print(suspect_SO)

# In[140]:


b1 = df_train.loc[df_train['ShipmentOrigin'].isin(suspect_SO)]

# In[141]:


b2 = b1.groupby(['PortOfLading']).size().reset_index(name='count')
lb = np.percentile(list(a1['count']), 25)
ub = np.percentile(list(a1['count']), 90)
b3 = b2.loc[(b2['count'] >= lb) & (b2['count'] <= ub)]
b4 = b3.sample(frac=0.33)
suspect_PoL = list(b4['PortOfLading'])

# In[142]:


z1 = df_train.loc[df_train['PortOfLading'].isin(suspect_PoL) & df_train['ShipmentOrigin'].isin(suspect_SO)]
z2 = df_train.loc[df_train['PortOfLading'].isin(suspect_PoL) & ~df_train['ShipmentOrigin'].isin(suspect_SO)]

# In[143]:


C1 = list(set(z1['ConsigneePanjivaID']).intersection(z2['ConsigneePanjivaID']))

C2 = df_train.loc[df_train['ConsigneePanjivaID'].isin(C1)].groupby(['ConsigneePanjivaID']).size().reset_index(
    name='count')



lb = np.percentile(list(C2['count']), 20)
ub = np.percentile(list(C2['count']), 90)
C3 = C2.loc[(C2['count'] >= lb) & (C2['count'] <= ub)]
suspect_C = list(C3['ConsigneePanjivaID'])
len(suspect_C)


q1 = df_test.loc[df_test['ShipmentOrigin'].isin(suspect_SO) & df_test['ConsigneePanjivaID'].isin(suspect_C)]


q2 = df_test.loc[df_test['PortOfLading'].isin(suspect_PoL) & df_test['ConsigneePanjivaID'].isin(suspect_C)]


cand_S = set(df_train.loc[df_train['ConsigneePanjivaID'].isin(suspect_C)]['ShipperPanjivaID'])
len(cand_S), len(suspect_C)


h1 = df_train.loc[df_train['ShipmentOrigin'].isin(suspect_SO) & df_train['PortOfLading'].isin(suspect_PoL)]
h2 = h1.groupby(['HSCode']).size().reset_index(name='count')
lb = np.percentile(list(C2['count']), 40)
ub = np.percentile(list(C2['count']), 90)
h3 = h2.loc[(h2['count'] >= lb) & (h2['count'] <= ub)].sample(frac=0.6)
suspect_H = list(h3['HSCode'])


q3 = df_test[
    df_test['ShipperPanjivaID'].isin(cand_S) & df_test['HSCode'].isin(suspect_H) &
    (df_test['ShipmentOrigin'].isin(suspect_SO) | df_test['PortOfLading'].isin(suspect_PoL))
    ]

fraud_samples = q1.append(q2, ignore_index=True).append(q3, ignore_index=True).drop_duplicates()
fraud_ids = list(fraud_samples[id_col])
non_fraud_samples = df_test.loc[~df_test[id_col].isin(fraud_ids)]



_fixed_set = ['HSCode', 'ConsigneePanjivaID', 'ShipmentOrigin', 'PortOfLading', 'ShipperPanjivaID']
_perturb_set = [_ for _ in domain_dims.keys() if _ not in _fixed_set]

pandarallel.initialize()

non_fraud_samples = non_fraud_samples.sample(50000)
fraud_samples = fraud_samples.sample(5000)

result_1 = fraud_samples.parallel_apply(
    generate_by_criteria,
    axis=1,
    args=(101, _fixed_set, _perturb_set, 0.5)
)



_fixed_set = ['HSCode', 'ConsigneePanjivaID', 'ShipperPanjivaID']
_perturb_set = [_ for _ in domain_dims.keys() if _ not in _fixed_set]

result_2 = non_fraud_samples.apply(
    generate_by_criteria,
    axis=1,
    args=(201, _fixed_set, _perturb_set, 0.2)
)


def add_anomaly_flag(row):
    global id_col
    val = str(int(row[id_col]))[-6:]
    if val == '010101':
        return True
    elif val == '010201':
        return True
    else:
        return False


result_1['anomaly'] = result_1.parallel_apply(add_anomaly_flag, axis=1)
result_2['anomaly'] = result_2.parallel_apply(add_anomaly_flag, axis=1)
# result_1['fraud'] = True
# result_2['fraud'] = False


df_1 = result_2.loc[result_2[anomaly_flag_col]==True]
df_2 = result_2.loc[result_2[anomaly_flag_col]==False]
try:
    del df_1[anomaly_flag_col]
    del df_2[anomaly_flag_col]
except:
    pass

file_a = os.path.join(DATA_DIR, DIR, 'anomalies_Fraud.csv')
file_na = os.path.join(DATA_DIR, DIR, 'anomalies_NotFraud.csv')
file_t = os.path.join(DATA_DIR, DIR,'test_data_v2.csv')
result_1.to_csv(file_a)
df_1.to_csv(file_na)
df_2.to_csv(file_t)

# def set_ground_truth(val):
#     if val:
#         return 1
#     else:
#         return 0
#
#
# # set ground truths
# df[ground_truth_col] = df[fraud_flag_col].parallel_apply(
#     set_ground_truth
# )
# # Valid labels are 0 and 1
# df[label_col] = -1
# df_s = df.sort_values(by='score')




