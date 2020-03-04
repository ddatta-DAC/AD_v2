#!/usr/bin/env python

import pandas as pd
import os
import numpy as np
import sys

sys.path.append('./..')
sys.path.append('./../..')
import yaml
import re
import pickle
import argparse
from collections import Counter
from operator import itemgetter
import glob
from pandarallel import pandarallel

pandarallel.initialize()
try:
    from . import utils_preprocess as utils
except:
    import utils_preprocess as utils

# --------------------------------------------------------------------- #

CONFIG_FILE = 'config_preprocessor_v02.yaml'
id_col = 'PanjivaRecordID'
save_dir = None
CONFIG = None
DIR = None
DATA_DIR = None
DIR_LOC = None


# ===================================================================== #

def set_up_config(_DIR):
    global CONFIG_FILE
    global DIR
    global save_dir
    global CONFIG
    global DATA_DIR


    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)
    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    DIR_LOC = re.sub('[0-9]', '', DIR)
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    DATA_DIR = save_dir
    return


set_up_config(DIR)


# =============================================== #

def get_data():
    global id_col
    global DATA_DIR
    f_name_train = 'train_data.csv'
    f_name_test = 'test_data.csv'

    df_train = pd.read_csv(os.path.join(DATA_DIR, f_name_train), index_col=None)
    df_test = pd.read_csv(os.path.join(DATA_DIR, f_name_test), index_col=None)

    with open(os.path.join(DATA_DIR, 'domain_dims.pkl'), 'rb') as fh:
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
        co_occurrence_dict,
        ref_df,
        id_col='PanjivaRecordID'
    ):
    print('  >> ', row)
    is_duplicate = True
    trials = 0
    max_trials = 100

    while is_duplicate:
        trials += 1
        p_d = np.random.choice(_perturb, size=2, replace=False)
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
        hash_val = utils.get_hash_aux(new_row, id_col)
        is_duplicate = utils.is_duplicate(ref_df, hash_val)

        if trials == max_trials:
            print('Error')
            new_row[_perturb[0]] = None
            break

    suffix = '00' + str(criteria)
    new_row[id_col] = utils.aux_modify_id(new_row[id_col], suffix)
    print(criteria, ' :: generated record')
    return new_row


'''
Main processing function
    ------------- Criteria 1 -------------------
    ## We define interesting records as ones which satisfy these 2 conditions:
    ### 1. contains both these companies
    ### 2. contains the route ( 'PortOfLading','PortOfUnlading' )
    ---------------------------------------------
    ------------- Criteria 2 --------------------
    ## Use the set of consignee and shippers obtained earlier
    ## We define interesting records as ones which satisfy these 2 conditions:
    ### 1. contain one of the companies
    ### 2. contain the triplet (ShipmentOrigin	HSCode	ShipmentDestination)
    ----------------------------------------------
'''


def main_process():
    df_train, df_test, domain_dims = get_data()
    columnWise_coOccMatrix_dict = utils.get_coOccMatrix_dict(df_train, id_col='PanjivaRecordID')

    # Select pairs of ports such that their count in (20,80) percentile
    # Select 10 % of such pairs
    kk = df_train.groupby(['PortOfLading', 'PortOfUnlading']).size().reset_index(name='count')
    lb = np.percentile(list(kk['count']), 20)
    ub = np.percentile(list(kk['count']), 80)
    kk_1 = kk.loc[(kk['count'] >= lb) & (kk['count'] <= ub)]
    kk_2 = kk_1.sample(frac=0.10)
    kk_2 = kk_2.reset_index(drop=True)
    del kk_2['count']
    target_PortOfLading_PortOfUnlading = kk

    # We need list of companies trading in these routes
    pp = df_train.merge(
        target_PortOfLading_PortOfUnlading,
        on=['PortOfLading', 'PortOfUnlading'],
        how='inner'
    )
    # Now we have the list of Shippers and Consignee who do business with them are actually suspicious
    # Assumption these are comapnes that trade along the route described by ('PortOfLading', 'PortOfUnlading')

    _frac = 0.10
    candidate_Shipper = list(set(pp['ShipperPanjivaID']))
    _count1 = int(_frac * domain_dims['ShipperPanjivaID'])
    _count2 = int(_frac * domain_dims['ConsigneePanjivaID'])

    target_Shipper = np.random.choice(candidate_Shipper, size=_count1, replace=False)
    pp_1 = pp.loc[pp['ShipperPanjivaID'].isin(target_Shipper)]

    candidate_Shipper = list(set(pp_1['ConsigneePanjivaID']))
    target_Consignee = np.random.choice(candidate_Shipper, size=_count2, replace=False)
    print('Number of interesting shippers ', len(target_Shipper))
    print('Number of interesting consignee ', len(target_Consignee))

    qq = df_train.loc[
        (df_train['ShipperPanjivaID'].isin(target_Shipper)) &
        (df_train['ConsigneePanjivaID'].isin(target_Consignee))
        ]

    # Select triplet of ports such that their count in (10,80) percentile
    # Select 10 % of them
    _lb = 10
    _ub = 80
    _frac = 0.10
    qq_1 = qq.groupby(['ShipmentOrigin', 'HSCode', 'ShipmentDestination']).size().reset_index(name='count')
    lb = np.percentile(list(qq_1['count']), _lb)
    ub = np.percentile(list(qq_1['count']), _ub)
    _count = int(_frac * len(qq_1))

    target_Origin_HSCode_Dest = qq_1.loc[(qq_1['count'] >= lb) & (qq_1['count'] <= ub)].sample(n=_count)
    del target_Origin_HSCode_Dest['count']

    # =========================================
    # select 2 of _perturb domains
    # set them to random options :  such that the row does not occur in train or test
    # =========================================
    ref_df = df_test.copy()
    ref_df = ref_df.append(df_train)
    hash_ref_df = utils.add_hash(ref_df, id_col)

    # ================================================ #
    # C1 ::
    # target_Shipper
    # candidate_Shipper
    # target_PortOfLading_PortOfUnlading
    # ================================================ #

    a = df_train.merge(
        target_PortOfLading_PortOfUnlading,
        on=['PortOfLading', 'PortOfUnlading'],
        how='inner'
    )

    a = a.loc[a['ConsigneePanjivaID'].isin(target_Consignee)]
    a = a.loc[a['ShipperPanjivaID'].isin(target_Shipper)]

    print(' Candidate list len', len(a))
    _fixed_set = ['ConsigneePanjivaID', 'PortOfLading', 'PortOfUnlading', 'ShipperPanjivaID']
    _perturb_set = [_ for _ in list(domain_dims.keys()) if _ not in _fixed_set]


    res_criteria_1_1 = a.apply(
        generate_by_criteria,
        axis=1,
        args=(101, _fixed_set, _perturb_set, columnWise_coOccMatrix_dict, hash_ref_df)
    )

    # ================================================ #
    # C2 :
    # target_Shipper
    # candidate_Shipper
    # target_Origin_HSCode_Dest
    # ================================================ #

    a = df_train.merge(
        target_Origin_HSCode_Dest,
        on=['ShipmentOrigin', 'HSCode', 'ShipmentDestination'],
        how='inner'
    )
    print('  Candidate list len', len(a))

    a1 = a.loc[a['ConsigneePanjivaID'].isin(target_Consignee)]
    _fixed_set = ['ConsigneePanjivaID', 'ShipmentOrigin', 'HSCode', 'ShipmentDestination']
    _perturb_set = [_ for _ in list(domain_dims.keys()) if _ not in _fixed_set]

    res_criteria_2_1 = a1.parallel_apply(
        generate_by_criteria,
        axis=1,
        args=(201, _fixed_set, _perturb_set, columnWise_coOccMatrix_dict, hash_ref_df)
    )

    a2 = a.loc[a['ShipperPanjivaID'].isin(target_Shipper)]
    _fixed_set = ['ShipmentOrigin', 'HSCode', 'ShipmentDestination', 'ShipperPanjivaID']
    _perturb_set = [_ for _ in list(domain_dims.keys()) if _ not in _fixed_set]

    res_criteria_2_2 = a2.parallel_apply(
        generate_by_criteria,
        axis=1,
        args=(202, _fixed_set, _perturb_set, columnWise_coOccMatrix_dict, hash_ref_df)
    )

    res_df = pd.DataFrame(columns=list(df_test.columns))
    res_df = res_df.append(res_criteria_1_1, ignore_index=True)
    res_df = res_df.append(res_criteria_2_1, ignore_index=True)
    res_df = res_df.append(res_criteria_2_2, ignore_index=True)
    res_df = res_df.dropna()
    res_df.to_csv(
        os.path.join(DATA_DIR, 'anomalies_W_pattern_1.csv'), index=False
    )

    # ------ #
    # Save the dfs

    intermediate_data_loc = os.path.join(DATA_DIR, 'fraud_targets')
    if not os.path.exists(intermediate_data_loc):
        os.mkdir(intermediate_data_loc)

    tmp = pd.DataFrame(columns=['ShipperPanjivaID'])
    tmp['ShipperPanjivaID'] = target_Shipper
    f_name = os.path.join(intermediate_data_loc,'gen_fraud_Shipper.csv')
    tmp.to_csv(f_name, index=None)

    tmp = pd.DataFrame(columns=['ConsigneePanjivaID'])
    tmp['ConsigneePanjivaID'] = target_Consignee
    f_name = os.path.join(intermediate_data_loc, 'gen_fraud_Consignee.csv')
    tmp.to_csv(f_name, index=None)

    f_name = os.path.join(intermediate_data_loc, 'gen_fraud_PortOfLading_PortOfUnlading.csv')
    target_PortOfLading_PortOfUnlading.to_csv( f_name , index=None)
    f_name = os.path.join(intermediate_data_loc, 'gen_fraud_Origin_HSCode_Destination.csv')
    target_Origin_HSCode_Dest.to_csv(f_name, index=None)
    return

# ----------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default=None
)

args = parser.parse_args()
DIR = args.DIR
main_process()
