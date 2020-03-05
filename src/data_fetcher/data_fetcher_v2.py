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


# ========================================= #
# Anomaly data
# ========================================= #

def get_anomaly_data( DATA_DIR, DIR, anomaly_type=1) :

    anomaly_data_file_name = 'matrix_anomaly_x_type' + str(anomaly_type) + '.npy'
    anomaly_idList_file_name = 'matrix_anomaly_idList_type' + str(anomaly_type) + '.npy'
    anomaly_data_file = os.path.join(
        DATA_DIR,
        DIR,
        anomaly_data_file_name
    )
    with open(anomaly_data_file, 'rb') as fh:
        anomaly_x = np.load(fh,allow_pickle=True)

    anomaly_idList_data_file =  os.path.join(
        DATA_DIR,
        DIR,
        anomaly_idList_file_name
    )
    with open(anomaly_idList_data_file, 'rb') as fh:
        anomaly_idList = np.load(fh)

    return anomaly_x, anomaly_idList



# ========================================= #
# Data fetcher for models
# ========================================= #

def get_data_MEAD_train(
        DATA_DIR,
        DIR,
        anomaly_type=1
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

    anomaly_x, anomaly_idList = get_anomaly_data(DATA_DIR, DIR, anomaly_type)
    return train_x_pos, train_x_neg, test_x, test_idList, anomaly_x, anomaly_idList


def get_data_test(
        DATA_DIR,
        DIR,
        anomaly_type=1
):
    anomaly_x, anomaly_idList = get_anomaly_data(DATA_DIR, DIR, anomaly_type)
    return train_x_pos, train_x_neg, test_x, test_idList, anomaly_x, anomaly_idList





def get_data_v2(
        DATA_DIR,
        DIR
):
    with open(os.path.join(
            DATA_DIR,
            DIR,
            'domain_dims.pkl'
    ), 'rb') as fh:
        domain_dims = pickle.load(fh)


    train_x_pos_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_train_positive.pkl'
    )

    with open(train_x_pos_file, 'rb') as fh:
        train_x_pos = pickle.load(fh)

    train_x_neg_file = os.path.join(
        DATA_DIR,
        DIR,
        'ape_negative_samples.pkl'
    )

    with open(train_x_neg_file, 'rb') as fh:
        train_x_neg = pickle.load(fh)

    test_x_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_positive.pkl'
    )

    with open(test_x_file, 'rb') as fh:
        test_x = pickle.load(fh)

    anomaly_data_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_anomalies.pkl'
    )
    test_id_list_file = os.path.join(
        DATA_DIR,
        DIR,
        'test_idList.pkl'
    )

    with open(anomaly_data_file, 'rb') as fh:
        anomaly_data = pickle.load(fh)

    with open(test_id_list_file, 'rb') as fh:
        _id_list = pickle.load(fh)
        test_anomaly_idList = _id_list[0]
        test_normal_idList = _id_list[1]

    test_pos = [test_normal_idList, test_x]
    test_anomaly = [test_anomaly_idList, anomaly_data]
    return train_x_pos, train_x_neg, test_pos, test_anomaly , domain_dims



def get_data_v3(
        DATA_DIR,
        DIR,
        c = 3
):
    with open(os.path.join(
            DATA_DIR,
            DIR,
            'domain_dims.pkl'
    ), 'rb') as fh:
        domain_dims = pickle.load(fh)


    train_x_pos_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_train_positive_v1.pkl'
    )

    with open(train_x_pos_file, 'rb') as fh:
        train_x_pos = pickle.load(fh)

    train_x_neg_file = os.path.join(
        DATA_DIR,
        DIR,
        'negative_samples_v1.pkl'
    )

    with open(train_x_neg_file, 'rb') as fh:
        train_x_neg = pickle.load(fh)
        train_x_neg = train_x_neg

    test_x_file = os.path.join(
        DATA_DIR,
        DIR,
        'matrix_test_positive.pkl'
    )

    with open(test_x_file, 'rb') as fh:
        test_x = pickle.load(fh)

    anomaly_data_file_f_name = 'matrix_test_anomalies_c' + str(c) + '.pkl'
    anomaly_data_file = os.path.join(
        DATA_DIR,
        DIR,
        anomaly_data_file_f_name
    )

    test_id_list_f_name = 'test_idList_c' + str(c) + '.pkl'
    test_id_list_file = os.path.join(
        DATA_DIR,
        DIR,
        test_id_list_f_name
    )
    print(' >> ', anomaly_data_file_f_name, test_id_list_f_name)

    with open(anomaly_data_file, 'rb') as fh:
        anomaly_data = pickle.load(fh)

    with open(test_id_list_file, 'rb') as fh:
        _id_list = pickle.load(fh)
        test_anomaly_idList = _id_list[0]
        test_normal_idList = _id_list[1]

    test_pos = [test_normal_idList, test_x]
    test_anomaly = [test_anomaly_idList, anomaly_data]
    return train_x_pos, train_x_neg, test_pos, test_anomaly , domain_dims
