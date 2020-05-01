import pickle
import os
import glob
import numpy as np
import pandas as pd


# ---------------------------------- #
# Standard data fetcher for all models
# ---------------------------------- #


def get_domain_dims(DATA_DIR, DIR):
    dd_file_path = os.path.join(DATA_DIR, DIR, 'domain_dims.pkl')
    with open(dd_file_path, 'rb') as fh:
        domain_dims = pickle.load(fh)
    _tmpDF = pd.DataFrame.from_dict(domain_dims, orient='index')
    _tmpDF = _tmpDF.reset_index()
    _tmpDF = _tmpDF.rename(columns={'index': 'domain'})
    _tmpDF = _tmpDF.sort_values(by=['domain'])
    res = {k: v for k, v in zip(_tmpDF['domain'], _tmpDF[0])}
    return res


# -------------

def get_train_x_csv(DATA_DIR, DIR):

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
# ratio :: percentage of data as "fraud"
# ========================================================== #
def get_anomaly_data_matrices(
        DATA_DIR,
        DIR,
        discriminate=True,
        ratio=None
):
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

    anomaly_F_x = np.load(anomaly_data_file_F, allow_pickle=True)
    anomaly_NF_x = np.load(anomaly_data_file_NF, allow_pickle=True)

    anomaly_idList_F_file = os.path.join(
        DATA_DIR,
        DIR,
        anomaly_idList_file_name_F
    )

    anomaly_idList_NF_file = os.path.join(
        DATA_DIR,
        DIR,
        anomaly_idList_file_name_NF
    )

    anomaly_idList_F = np.load(anomaly_idList_F_file)
    anomaly_idList_NF = np.load(anomaly_idList_NF_file)
    # -----------
    #
    # -----------
    if ratio is not None:
        F_count = anomaly_idList_F.shape[0]
        NF_count = anomaly_idList_NF.shape[0]

        idx_F = list(np.arange(F_count))
        np.random.shuffle(idx_F)
        anomaly_idList_F = anomaly_idList_F[idx_F]
        anomaly_F_x = anomaly_F_x[idx_F]

        idx_NF = list(np.arange(NF_count))
        np.random.shuffle(idx_NF)
        anomaly_idList_NF = anomaly_idList_NF[idx_NF]
        anomaly_NF_x = anomaly_NF_x[idx_NF]

        # Ensure ratio of fraud to non fraud maintained
        if float(F_count) / (NF_count + F_count) <= ratio:
            nfc = int((1 - ratio) / ratio * F_count)
            anomaly_idList_NF = anomaly_idList_NF[:nfc]
            anomaly_NF_x = anomaly_NF_x[:nfc]

        else:
            fc = int(ratio / (1 - ratio) * NF_count)
            anomaly_F_x = anomaly_F_x[fc]
            anomaly_idList_F = anomaly_idList_F[fc]

        assert (abs(
            float(F_count) / (NF_count + F_count) - ratio) > 0.025), "Ratio of Fraud & non Frauds not maintained!! "

    if discriminate:
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


# ---------------------------
# Function to get test data
# This is test dat for (older) AD system.
# equal number of Anomaly and non-Anomalies
# ---------------------------

def get_data_test(
        DATA_DIR,
        DIR
):
    _, test_x, test_idList = get_data_base_x(DATA_DIR, DIR)
    anomaly_x, anomaly_idList = get_anomaly_data_matrices(
        DATA_DIR,
        DIR,
        discriminate=False
    )
    anomaly_count = len(anomaly_idList)
    normal_count = len(test_idList)

    _count = min(anomaly_count, normal_count)
    test_x = test_x[:_count]
    test_idList = test_idList[:_count]
    anomaly_x = anomaly_x[:_count]
    anomaly_idList = anomaly_idList[:_count]

    return test_x, test_idList, anomaly_x, anomaly_idList

# ------------------------------------------------------------- #

def get_testing_data_as_DF(
        DATA_DIR,
        DIR
):

    _, test_x, test_idList = get_data_base_x(DATA_DIR, DIR)
    anomaly_x , anomaly_idList = get_anomaly_data_matrices(
        DATA_DIR,
        DIR,
        discriminate=True,
        ratio=0.5
    )
    anomaly_F_x = anomaly_x[0]
    anomaly_NF_x = anomaly_x[1]

    anomaly_idList_F = anomaly_idList[0]
    anomaly_idList_NF = anomaly_idList[1]

    tmp_df = get_train_x_csv(
        DATA_DIR,
        DIR
    )
    cols = list(tmp_df.columns)

    _df1 = pd.DataFrame(
        data = np.hstack([np.reshape(test_idList,[-1,1]), test_x]),
        columns = cols
    )
    _df1['anomaly'] = False
    _df1['fraud'] = False

    _df2 = pd.DataFrame(
        data=np.hstack([np.reshape(anomaly_idList_F, [-1, 1]), anomaly_F_x]),
        columns=cols
    )
    _df2['anomaly'] = True
    _df2['fraud'] = True

    _df3 = pd.DataFrame(
        data=np.hstack([np.reshape(anomaly_idList_NF, [-1, 1]), anomaly_NF_x]),
        columns=cols
    )

    _df3['anomaly'] = True
    _df3['fraud'] = False

    if len(_df1) >  len(_df2) + len(_df3):
        _df1 = _df1.sample(
            len(_df2) + len(_df3)
        )
        DF = _df1.copy()
        DF = DF.append(_df2, ignore_index=True)
        DF = DF.append(_df3, ignore_index=True)
    else :
        DF = _df2.copy()
        DF = DF.append(_df3, ignore_index=True)
        DF = DF.sample(len(_df1))
        DF = DF.append(_df1, ignore_index=True)

    return DF


# -----------------------------------------------------------------------
# Function to obtain data for stage 2
#
# -----------------------------------------------------------------------
def get_Stage2_data_as_DF(
        DATA_DIR,
        DIR,
        id_col = 'PanjivaRecordID',
        fraud_ratio = 0.5,
        anomaly_ratio=0.4,
        total_size = 10000
):

    anomalies_NotFraud_file_name = 'anomalies_NotFraud.csv'
    anomalies_Fraud_file_name = 'anomalies_Fraud.csv'
    normal_data_file_name =  'test_data_v2.csv'

    anomalies_F_df = pd.read_csv(
        os.path.join(
            DATA_DIR, DIR, anomalies_Fraud_file_name
        )
    )
    anomalies_F_df['anomaly'] = True
    anomalies_F_df['fraud'] = True

    anomalies_NF_df = pd.read_csv(
        os.path.join(
            DATA_DIR, DIR, anomalies_NotFraud_file_name
        )
    )
    anomalies_NF_df['anomaly']  = True
    anomalies_NF_df['fraud'] = False

    normal_df = pd.read_csv(
        os.path.join(
            DATA_DIR, DIR, normal_data_file_name
        )
    )

    normal_df['anomaly'] = False
    normal_df['fraud'] = False

    res_df = normal_df.copy()
    res_df = res_df.append(anomalies_NF_df, ignore_index=True)
    res_df = res_df.append(anomalies_F_df, ignore_index=True)
    return res_df


    R2 = anomaly_ratio
    R1 = fraud_ratio

    if R1 * R2 * total_size > len(anomalies_F_df):
       print('[ERROR] Check data size and Ratios')
       exit(1)

    #-------------------------
    # Adjust total size so as to keep the ratio intact but the
    # -------------
    # a = # of Fraud
    # b = # non Fraud
    # c = # normal
    #  a = (a + b) * fraud_ratio
    # (a + b) = (a + b + c) * anomaly_ratio
    #  If max given
    #       a + b + c = max
    #  else :
    #       Use entire normal set
    # ---------------

    R2 = anomaly_ratio
    R1 = fraud_ratio

    c  = int( (1 - R2) * total_size )
    M = int(R2 * total_size)
    a = int(M * R1)
    b = int(M * (1-R1))

    F = anomalies_F_df.sample(n = a)
    NF = anomalies_NF_df.sample(n = b )
    P = normal_df.sample (n = c)
    res_df = pd.DataFrame( P )
    res_df = res_df.append(F, ignore_index=True)
    res_df = res_df.append(NF, ignore_index=True)
    return res_df

#-------------------------------------------





