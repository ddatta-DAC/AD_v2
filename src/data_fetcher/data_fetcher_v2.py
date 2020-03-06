import pickle
import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------- #
# Standard data fetcher for all models
# ---------------------------------- #


def get_domain_dims(DATA_DIR, DIR):
    dd_file_path = os.path.join( DATA_DIR, DIR, 'domain_dims.pkl')
    with open(dd_file_path, 'rb') as fh:
        domain_dims = pickle.load(fh)
    _tmpDF = pd.DataFrame.from_dict(domain_dims, orient='index')
    _tmpDF = _tmpDF.reset_index()
    _tmpDF = _tmpDF.rename(columns={'index': 'domain'})
    _tmpDF = _tmpDF.sort_values(by=['domain'])
    res = {k: v for k, v in zip(_tmpDF['domain'], _tmpDF[0])}
    return res
# -------------

def get_train_x_csv(DATA_DIR,DIR):
    fp = os.path.join(
        DATA_DIR,
        DIR,
        'train_data.csv'
    )
    df = pd.read_csv(fp, index_col=None)
    return df


# -------------
def get_data_base_x(
        DATA_DIR,
        DIR
):

    train_x_pos = None
    test_x = None

    try:
        train_x_pos_file = os.path.join(
            DATA_DIR,
            DIR,
            'train_matrix_x_positive.npy'
        )

        with open(train_x_pos_file, 'rb') as fh:
            train_x_pos = np.load(fh)
    except:
        print('Error reading file ::', train_x_pos_file)

    try:
        test_x_file = os.path.join(
            DATA_DIR,
            DIR,
            'test_matrix_x_positive.npy'
        )

        with open(test_x_file, 'rb') as fh:
            test_x = np.load(fh)
    except:
        print('Error reading file ::', test_x_file)

    # -------------
    # Test set id _list
    # -------------
    test_idList = None
    try:
        test_idList_file = os.path.join(
            DATA_DIR,
            DIR,
            'test_idList.npy'
        )

        with open(test_idList_file, 'rb') as fh:
            test_idList = np.load(fh)
    except:
        print('Error reading file ::', test_idList_file)

    return train_x_pos, test_x, test_idList


# ========================================================== #
# Anomaly data
# ========================================================== #
def get_anomaly_data( DATA_DIR, DIR, discriminate=False) :

    anomaly_data_file_name_F = 'matrix_anomaly_F.npy'
    anomaly_data_file_name_NF = 'matrix_anomaly_NF.npy'
    anomaly_idList_file_name_F = 'matrix_anomaly_idList_F.npy'
    anomaly_idList_file_name_NF = 'matrix_anomaly_idList_NF.npy'


    anomaly_data_file_F = os.path.join(
        DATA_DIR,
        DIR,
        anomaly_data_file_name_F
    )

    anomaly_data_file_NF = os.path.join(
        DATA_DIR,
        DIR,
        anomaly_data_file_name_NF
    )

    with open(anomaly_data_file_F, 'rb') as fh:
        anomaly_F_x = np.load(fh, allow_pickle=True)

    with open(anomaly_data_file_NF, 'rb') as fh:
        anomaly_NF_x = np.load(fh, allow_pickle=True)

    anomaly_idList_F_file =  os.path.join(
        DATA_DIR,
        DIR,
        anomaly_idList_file_name_F
    )

    anomaly_idList_NF_file = os.path.join(
        DATA_DIR,
        DIR,
        anomaly_idList_file_name_NF
    )

    with open(anomaly_idList_F_file, 'rb') as fh:
        anomaly_idList_F = np.load(fh)

    with open(anomaly_idList_NF_file, 'rb') as fh:
        anomaly_idList_NF = np.load(fh)
    if discriminate :
        return [anomaly_F_x, anomaly_NF_x], [anomaly_idList_F, anomaly_idList_NF]
    else:
        anomaly_x = np.vstack([anomaly_F_x, anomaly_NF_x])
        anomaly_idList = np.hstack([anomaly_idList_F, anomaly_idList_NF])
        return anomaly_x, anomaly_idList



# ========================================= #
# Data fetcher for models
# ========================================= #

def get_data_MEAD_train(
        DATA_DIR,
        DIR
):
    train_x_pos, test_x, test_idList = get_data_base_x(DATA_DIR, DIR)
    train_x_neg = None

    train_x_neg_file = os.path.join(
        DATA_DIR,
        DIR,
        'negative_samples_mead.npy'
    )
    try:
        with open(train_x_neg_file, 'rb') as fh:
            train_x_neg = np.load(fh)
    except:
        print('Error reading file ::', train_x_neg_file)

    return train_x_pos, train_x_neg


def get_data_test(
        DATA_DIR,
        DIR
):
    _, test_x, test_idList = get_data_base_x(DATA_DIR, DIR)
    anomaly_x, anomaly_idList = get_anomaly_data(
        DATA_DIR,
        DIR,
        discriminate=False
    )
    return test_x, test_idList, anomaly_x, anomaly_idList




