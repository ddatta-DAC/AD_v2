# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------


from multiprocessing import Pool
from numpy import load as load_np
from numpy import save as save_np
from scipy.linalg.blas import sgemm
from hashlib import md5
from scipy.sparse import csr_matrix
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
sys.path.append('.')
sys.path.append('./..')
sys.path.append('./../..')
import pickle
import logging
from datetime import datetime
import argparse
import multiprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score

from pandarallel import pandarallel
pandarallel.initialize()
import yaml
sys.path.append('./../..')
sys.path.append('./..')
from joblib import Parallel, delayed
from src.utils import coOccMatrixGenerator

try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher

# try:
#     from . import network_similarity as NS
# except:
#     import network_similarity as NS


# --------------------------------------------
# Global variables
# ----------------------------------------------


DIR = None
TARGET_DATA_SOURCE = './../../AD_system_output_v2'
CONFIG = None
config_file = 'config.yaml'
KNN_k = None
id_col = 'PanjivaRecordID'
data_max_size = None
nodeObj_dict_file = 'nodeObj_dict.pkl'
REFRESH_NODES = False
nodeObj_Dict = None
model_use_data_DIR = None
list_MP_OBJ = None
domain_dims = None
KNN_dir = None
Logging_Dir = 'Log'
# ------------------------------------------------------------
# ---------         First function to be executed
# Call this to set up global variables
# ------------------------------------------------------------

SOURCE_DATA_DIR_1 = './../../generated_data_v2'

def setup():
    global DIR
    global config_file
    global model_use_data_DIR
    global TARGET_DATA_SOURCE
    global domain_dims
    global KNN_k
    global data_max_size
    global KNN_dir
    global CONFIG

    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)

    data_max_size = CONFIG['data_max_size']
    KNN_k = CONFIG['KNN_k']
    model_use_data_DIR = CONFIG['model_use_data_DIR']
    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    if not os.path.exists(os.path.join(model_use_data_DIR, DIR)):
        os.mkdir(os.path.join(model_use_data_DIR, DIR))
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    domain_dims = get_domain_dims(DIR)
    KNN_dir = 'KNN'+ '_' + str(KNN_k)+ '_' + str(data_max_size)
    KNN_dir = os.path.join(model_use_data_DIR, KNN_dir)
    return

# ---------------------------------------

def get_logger():
    global Logging_Dir
    global DIR
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    OP_DIR = os.path.join(Logging_Dir, DIR)
    log_file = 'results.log'
    if not os.path.exists(Logging_Dir):
        os.mkdir(Logging_Dir)

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)
    log_file_path = os.path.join(OP_DIR, log_file)
    handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(str(datetime.utcnow()))
    return logger

def close_logger(logger):

    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    return


def get_training_data(DIR):
    global SOURCE_DATA_DIR_1
    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR_1, DIR)
    return data


def get_domain_dims(DIR):
    with open(
            os.path.join(
                SOURCE_DATA_DIR_1,
                DIR,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


# -----------------------------------------------
def matrix_multiply(
        matrix_list,
        _csr=True

):
    list_dimensions = [matrix_list[0].shape[0], matrix_list[0].shape[1]]
    for _ in matrix_list[1:]:
        list_dimensions.append(_.shape[1])

    def optimal_parenthesization(a):
        n = len(a) - 1
        u = [None] * n
        u[0] = [[None, 0]] * n
        for j in range(1, n):
            v = [None] * (n - j)
            for i in range(n - j):
                m = None
                for k in range(j):
                    s1, c1 = u[k][i]
                    s2, c2 = u[j - k - 1][i + k + 1]
                    c = c1 + c2 + a[i] * a[i + k + 1] * a[i + j + 1]
                    if m is None or c < m:
                        s = k
                        m = c
                v[i] = [s, m]
            u[j] = v

        def aux(i, j):
            s, c = u[j][i]
            if s is None:
                return i
            else:
                return [aux(i, s), aux(i + s + 1, j - s - 1)]

        return aux(0, n - 1)

    def aux_mul(a, b):
        if _csr:
            return a * b
        else:
            return np.matmul(a, b)

    def mult(_list_indices):

        if type(_list_indices[0]) == int and type(_list_indices[1]) == int:
            mat_a = matrix_list[_list_indices[0]]
            mat_b = matrix_list[_list_indices[1]]
            return aux_mul(mat_a, mat_b)
        elif type(_list_indices[0]) == int and type(_list_indices[1]) == list:
            mat_a = matrix_list[_list_indices[0]]
            mat_b = mult(_list_indices[1])
            return aux_mul(mat_a, mat_b)
        elif type(_list_indices[0]) == list and type(_list_indices[1]) == int:
            mat_a = mult(_list_indices[0])
            mat_b = matrix_list[_list_indices[1]]
            return aux_mul(mat_a, mat_b)
        else:
            mat_a = mult(_list_indices[0])
            mat_b = mult(_list_indices[1])
            return aux_mul(mat_a, mat_b)

    mult_order = optimal_parenthesization(list_dimensions)
    return mult(mult_order)


# -------------------------------------------------------------------
def get_transition_matrix(domain1, domain2):
    global coOccDict
    if domain1 < domain2:
        key = domain1 + '_+_' + domain2
        return coOccDict[key]
    else:
        key = domain2 + '_+_' + domain1
        return np.transpose(coOccDict[key])


# ---------------------------------------------------------------------
# Read in the meta paths
# ---------------------------------------------------------------------
def get_metapath_list():
    MP_list = []
    with open('metapaths.txt', 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            line = line.strip()
            _list = line.split(',')
            _list = [_.strip() for _ in _list]
            MP_list.append(_list)
    return MP_list


# -----------------------------------------------------------------------

class MP_object:
    id = 0

    @staticmethod
    def assign_id():
        t = MP_object.id
        MP_object.id = MP_object.id + 1
        return t

    @staticmethod
    def get_signature(MP_list):
        _signature = ''.join([''.join(_) for _ in MP_list])
        signature = str(md5(str.encode(_signature)).hexdigest())
        return signature

    @staticmethod
    def GET_mp_obj(MP):
        global model_use_data_DIR
        signature = MP_object.get_signature(MP)
        print(signature)
        saved_file_name = 'mp_object_' + signature + '.pkl'
        saved_file_path = os.path.join(
            model_use_data_DIR,
            saved_file_name
        )

        if os.path.exists(saved_file_path):
            print('File exists!')
            with open(saved_file_path, "rb") as fh:
                obj = pickle.load(fh)
            return obj
        else:
            obj = MP_object(MP)
            with open(saved_file_path, 'wb') as fh:
                pickle.dump(
                    obj,
                    fh,
                    pickle.HIGHEST_PROTOCOL
                )
            return obj

    def __init__(self, MP):
        global domain_dims

        self.mp = MP + MP[::-1][1:]
        self.id = MP_object.assign_id()
        self.simMatrix = None
        # set up the commuting matrix for PathSim
        matrix_list = []

        for r1, r2 in zip(self.mp[:-1], self.mp[1:]):
            mat = get_transition_matrix(r1, r2)
            mat = csr_matrix(mat)
            matrix_list.append(mat)

        self.CM = matrix_multiply(matrix_list)
        return

    def calc_PathSim(
            self,
            t_df,
            domain_dims
    ):
        global id_col
        global model_use_data_DIR
        f_name = 'sim_matrix_mp_' + str(self.id) + '.npy'

        t_df = t_df.sort_values(by=[id_col], ascending=True)
        n = len(t_df)
        simMatrix_path = os.path.join(
            model_use_data_DIR,
            f_name
        )
        self.simMatrix = None

        if os.path.exists(simMatrix_path):
            simMatrix = load_np(simMatrix_path)
            print(simMatrix.shape)
            self.simMatrix = simMatrix
        else:
            print('MetaPath :', self.mp)
            conn_domain = self.mp[0]

            A_t_d = np.zeros([n, domain_dims[conn_domain]])
            d_vals = list(t_df[conn_domain])
            A_t_d[np.arange(n), d_vals] = 1

            # A_t_d = csr_matrix(A_t_d)
            P2 = sgemm(1, self.CM.toarray(), A_t_d.transpose())
            _simMatrix = sgemm(1, A_t_d, P2)

            D = _simMatrix.diagonal()
            simMatrix = _simMatrix.copy()

            args = [(i, j) for i, j in zip(*_simMatrix.nonzero())]
            n_jobs = multiprocessing.cpu_count()

            # Parallelize !!
            def aux1(i, j):
                v_d = (D[i] + D[j])
                if v_d == 0:
                    simMatrix[i, j] = 0
                    return
                val = 2 * _simMatrix[i, j] / v_d
                simMatrix[i, j] = val
                return

            _ = Parallel(
                n_jobs=n_jobs,
                require='sharedmem')(
                delayed(aux1)(i_j[0], i_j[1]) for i_j in args
            )
            self.simMatrix = simMatrix
            save_np(
                simMatrix_path,
                simMatrix
            )

        return


# -------------------------------------------------------------------
def network_creation(
        train_x_df,
        MP_list,
        id_col='PanjivaRecordID'
):
    global coOccDict
    global model_use_data_DIR

    coOccDict_file = os.path.join(model_use_data_DIR, 'coOccDict.pkl')
    if os.path.exists(coOccDict_file):
        with open(coOccDict_file, 'rb') as fh:
            coOccDict = pickle.load(fh)
    else:
        coOccDict = coOccMatrixGenerator.get_coOccMatrix_dict(train_x_df, id_col)
        with open(coOccDict_file, 'wb') as fh:
            pickle.dump(
                coOccDict,
                fh,
                pickle.HIGHEST_PROTOCOL
            )
    list_mp_obj = []
    for mp in MP_list:
        mp_obj = MP_object.GET_mp_obj(mp)
        list_mp_obj.append(mp_obj)

    return list_mp_obj


# ------------------------------------------ #
def set_up_closest_K_by_RecordID(
        args
):
    global list_MP_OBJ
    global id_col
    global model_use_data_DIR

    Record_ID = args[0]
    K =  args[1]
    save_Dir = args[2]

    f_name = str(Record_ID) + '.csv'
    f_path = os.path.join(save_Dir, f_name)

    R2S_df = record_2_serial_ID_df.copy()
    serialID = list(
        R2S_df.loc[R2S_df[id_col] == Record_ID]['Serial_ID']
    )[0]

    sim_values = []
    for mp_obj in list_MP_OBJ:

        sim_vals = mp_obj.simMatrix[serialID, :]
        sim_values.append(sim_vals)

    sim_values = np.vstack(sim_values)
    sim_values = np.median(sim_values, axis=0)
    res_df = pd.DataFrame(
        data=sim_values,
        columns=['score']
    )
    res_df = res_df.reset_index(drop=True)
    res_df['Serial_ID'] = res_df.index

    # place the id col
    def aux_place_Record_ID(row):
        sID = row['Serial_ID']
        return list(R2S_df.loc[R2S_df['Serial_ID'] == sID][id_col])[0]

    res_df[id_col] = res_df.apply(
        aux_place_Record_ID,
        axis=1
    )
    res_df = res_df.sort_values(by=['score'], ascending=False)
    res_df = res_df.head(K)
    res_df.to_csv(f_path, index=None)
    return


# ------------------------------------------ #

def aux_precompute_PathSimCalc( args ):
    mp_obj = args[0]
    _df = args[1]
    _dd = args[2]
    mp_obj.calc_PathSim(_df,_dd)
    return mp_obj

def process_target_data(
        target_df,
        _record_2_serial_ID_df,
        K=100
):
    global record_2_serial_ID_df
    global list_MP_OBJ
    global KNN_dir

    if not os.path.exists(KNN_dir):
        os.mkdir(KNN_dir)

    record_2_serial_ID_df = _record_2_serial_ID_df
    args = [(_obj , target_df.copy(), domain_dims.copy()) for _obj in list_MP_OBJ]
    n_jobs = multiprocessing.cpu_count()

    with Pool(n_jobs) as p:
        res = p.map(aux_precompute_PathSimCalc ,args)
    list_MP_OBJ = res



    n_jobs = multiprocessing.cpu_count()
    args = [(_record_ID, K,KNN_dir) for _record_ID in list(target_df[id_col])]
    with Pool(n_jobs) as p:
        res = p.map(set_up_closest_K_by_RecordID, args)
    return

def network_initialize():
    global list_MP_OBJ
    train_df = get_training_data(DIR)
    MP_list = get_metapath_list()
    list_mp_obj = network_creation(
        train_df,
        MP_list
    )
    list_MP_OBJ = list_mp_obj
    return




# --------------------------------------------------------- #
# Read data that is output of the Anomaly Detection system
# --------------------------------------------------------- #

def read_target_data():
    global DIR
    global TARGET_DATA_SOURCE
    global data_max_size
    global id_col
    global LOGGER

    csv_f_name = 'scored_test_data.csv'
    df = pd.read_csv(
        os.path.join(
            TARGET_DATA_SOURCE,
            DIR,
            csv_f_name), index_col=None
    )
    # ----------------
    # Check if previously target data has been read and calculations made
    # ----------------
    f_path =  os.path.join(
        model_use_data_DIR,
        'record_2_serial_ID.csv'
    )

    if os.path.exists( f_path):
        tmp_df = pd.read_csv(f_path, index_col=None)
        valid_ids = list(tmp_df[id_col])
        df = df.loc[df[id_col].isin(valid_ids)]
    else:
        df = df.sample(data_max_size)
    LOGGER.info('Length of data :: ' +  str(len(df)) )
    df = df.sort_values(
        by=['score']
    )
    return df


def get_record_2_serial_ID_df(target_df):
    global id_col
    global model_use_data_DIR
    record_2_serial_file = os.path.join(
        model_use_data_DIR,
        'record_2_serial_ID.csv'
    )

    if os.path.exists(record_2_serial_file):
        return pd.read_csv(
            record_2_serial_file, index_col=None
        )

    record_2_serial_ID = {
        e[1]: e[0] for e in enumerate(list(target_df[id_col]), 0)
    }

    record_2_serial_ID_df = pd.DataFrame(
        record_2_serial_ID.items(),
        columns=[id_col, 'Serial_ID']
    )

    record_2_serial_ID_df.to_csv(
        record_2_serial_file, index=False
    )
    return record_2_serial_ID_df

# -----------------------------------
# ---------------
# Algorithm ::
# Create a network
# Calculate SimRank between the Transaction nodes
#
# With partially labelled data - Train a classifier
# Classify points on the unlabelled data (transaction instances : where features are entities + anomaly scores )
# Set final label as
# Sign ( lambda * Weighted(similarity based) of labels of its K nearest (labelled) neighbors + (1-lambda) predicted label )
# ----------------

# ------------------------------------------
# Asssumption that input dataframe is sorted by scores (ascending)
# ------------------------------------------
def execute_iterative_classification(
    df,
    cur_checkpoint = 10
):
    global id_col
    global domain_dims
    global classifier_type
    global LOGGER
    global KNN_k

    label_col = 'y'
    epsilon = 0.20
    k = KNN_k

    LOGGER.info( " K = " + str(k))
    LOGGER.info("Length of data :: " + str(len(df)))
    LOGGER.info("Current percentage of data labelled  :: " + str(cur_checkpoint))
    def set_y(row):
        if row['fraud']:
            return 1  # labelled True
        elif not row['fraud']:
            return -1  # labelled false
        else:
            return 0  # unknown

    def update_label(row, ref_df, epsilon, _k):
        _id = row[id_col]
        _lambda = 0.5
        ref_df = ref_df[[id_col,label_col]]
        f_path = os.path.join(KNN_dir,str(_id) + '.csv')
        _sim_df = pd.read_csv(f_path,index_col=None)
        _sim_df = _sim_df.merge(
            ref_df,
            on=id_col,
            how ='left'
        )
        _sim_df = _sim_df.head(_k+1)
        _sim_df = _sim_df.tail(_k)
        _l = list(_sim_df[label_col])
        _s = list(_sim_df['score'])

        res = np.sum(np.multiply(_l,_s))/ np.sum(_s)
        res = _lambda * row[label_col] + (1-_lambda) * res
        if np.abs(res) >= epsilon :
            return np.sign(res)
        else :
            return 0

    # ---------------------------------------------------- #

    clf = None
    # Train initial Classifier model
    if classifier_type == 'RF':
        clf = RandomForestClassifier(
            n_estimators=50,
            n_jobs=-1,
            verbose=1,
            warm_start=False
        )
    elif classifier_type == 'SVM':
        clf = SVC(
            kernel='poly',
            degree='4'
        )

    df_master = df.copy()
    record_count = len(df)

    # count of how many labelled and unlabelled data points
    l_count = int(record_count * cur_checkpoint / 100)
    u_count = record_count - l_count
    one_hot_columns = list(domain_dims.keys())

    df = pd.get_dummies(
        df,
        columns=one_hot_columns
    )
    df_L = df.head(l_count).copy()
    df_U = df.tail(u_count).copy()

    labelled_instance_ids = list(df_L[id_col])
    unlabelled_instance_ids = list(df_U[id_col])

    df_iter = df_master.copy()
    df_iter_L = df_iter.loc[df_iter[id_col].isin(labelled_instance_ids)]
    df_iter_U = df_iter.loc[df_iter[id_col].isin(unlabelled_instance_ids)]

    df_iter_L[label_col] = df_iter_L.parallel_apply(
        set_y,
        axis=1
    )
    fixed_values_df = pd.DataFrame(df_iter_L[[id_col,label_col]],copy=True)
    df_iter_U[label_col] = 0
    df_iter = df_iter_L.append(df_iter_U,ignore_index=True)

    clf_train_df = df_L.copy()

    clf_train_df[label_col] = clf_train_df.parallel_apply(
        set_y,
        axis=1
    )
    Y = list(clf_train_df[label_col])

    remove_cols = [label_col,'anomaly','fraud',id_col]
    for rc in remove_cols:
        try:
            del clf_train_df[rc]
        except:
            pass

    X = clf_train_df.values
    clf.fit(X,Y)
    clf_test_df = df_U.copy()
    for rc in remove_cols:
        try:
            del clf_test_df[rc]
        except:
            pass

    X_test = clf_test_df.values
    Y_pred = clf.predict(X_test)
    Y_pred = np.reshape(Y_pred,-1)
    clf_test_df[label_col] = Y_pred
    df_iter_U[label_col] = Y_pred

    # ----------------------------------------------
    # Obtain updated label
    # lv = Neighbor similarity_score * label
    # Updated label = Sign(lv) if |lv| > epsilon
    # ----------------------------------------------
    pandarallel.initialize()
    df_iter_U[label_col] = df_iter_U.parallel_apply(
        update_label,
        axis=1,
        args=(df_iter.copy(), epsilon, k,)
    )
    df_iter = df_iter_L.append(df_iter_U, ignore_index=True)

    num_iter = 0
    max_iter = 25

    while num_iter < max_iter:
        zero_count = len(df_iter.loc[df_iter[label_col] == 0])
        if zero_count == 0:
            print('Breaking ...')
            break

        num_iter +=1
        print(' Iteration :: ', num_iter)
        # now re train classifier
        new_clf_df = pd.get_dummies(
            df_iter.copy(),
            columns=one_hot_columns
        )

        new_known_label_ids = list(
            df_iter.loc[df_iter[label_col]!=0][id_col]
        )

        new_clf_train_df = new_clf_df.loc[
            new_clf_df[id_col].isin(new_known_label_ids)
        ]
        new_clf_test_df = new_clf_df.loc[
            ~new_clf_df[id_col].isin(new_known_label_ids)
        ]
        new_train_Y = list(new_clf_train_df[label_col])
        new_unknown_label_ids = list(new_clf_test_df[id_col])

        for rc in remove_cols:
            try:
                del new_clf_train_df[rc]
                del new_clf_test_df[rc]
            except:
                pass


        new_train_X = new_clf_train_df.values
        new_test_X  = new_clf_test_df.values

        clf.fit(new_train_X,new_train_Y)
        new_pred_Y = clf.predict(new_test_X)
        new_pred_Y = np.reshape(new_pred_Y,-1)

        updater_df = pd.DataFrame(
            data = np.stack([new_unknown_label_ids, new_pred_Y],axis=1),
            columns = [id_col, label_col]
        )

        # ----------------------------------
        # Assign new labels in df_iter
        # ---------------------------------
        def func_assign(_row, ref_df):
            row = _row.copy()
            _res = list(
                ref_df.loc[ref_df[id_col] == row[id_col]][label_col])
            if len(_res) > 0:
                row[label_col] = _res[0]
            return row

        # --------------------------
        # Update df_iter
        # --------------------------
        # Place newly predicted (clf) labels
        pandarallel.initialize()
        df_iter = df_iter.parallel_apply(
            func_assign,
            axis=1,
            args=(updater_df,)
        )
        print(" Placed newly predicted (clf) labels ")
        # Use the currently predicted and known labels to update the labels("unknown")
        _ref_df = df_iter.copy()

        df_iter[label_col] = df_iter.parallel_apply(
            update_label,
            axis=1,
            args=(_ref_df, epsilon, k,)
        )
        print(" Updated labels using KNN ")

        # clamp down original known labels

        df_iter = df_iter.parallel_apply(
            func_assign,
            axis=1,
            args=(fixed_values_df,)
        )
        print(" Placed back original known labels ")
        # update epsilon
        epsilon = max(0.05, epsilon * 0.75)

    print ('Starting evaluation ')

    # ------- Evaluation -------- #
    true_label_name = 'y_true'

    def place_true_labels(val):
        if val :
            return 1
        else:
            return -1

    df_eval = pd.DataFrame(
        df_iter.loc[df_iter[id_col].isin(unlabelled_instance_ids)],
        copy=True
    )

    for dd in domain_dims.keys():
        del df_eval[dd]

    df_eval[true_label_name] = df_eval['fraud'].parallel_apply(
        place_true_labels
    )

    # df_eval1 = pd.DataFrame(
    #     df_eval.sort_values(by=[label_col,'score'], ascending=[False,True]),
    #     copy=True
    # )

    df_eval1 = pd.DataFrame(
        df_eval.sort_values(by=['score'], ascending=True),
        copy=True
    )

    df_eval2 = pd.DataFrame(
        df_eval.sort_values(by=['score'], ascending=True),
        copy=True
    )
    results_print = pd.DataFrame(
        columns=['next %', 'precision', 'recall', 'f1', 'balanced_accuracy']
    )

    for point in [10, 20, 30, 40, 50]:
        _count = int(len(df_master) * point / 100)
        df_tmp = df_eval1.head(_count)
        y_true = list(df_tmp[true_label_name])
        y_pred = list(df_tmp[label_col])

        # Calculate precision and recall

        precision = precision_score(y_true, y_pred, labels=[-1,1], pos_label=1)
        recall = recall_score(y_true, y_pred, labels=None, pos_label=1)
        f1 = 2 * precision * recall / (precision + recall)
        b_acc = balanced_accuracy_score(y_true, y_pred)
        _dict = {
         'next %' : point,
         'precision' : round(precision,3),
         'recall' : round(recall,3),
         'f1' : round(f1,3),
         'balanced_accuracy': round(b_acc,3)
        }

        results_print = results_print.append(_dict, ignore_index=True)


        # accuracy = round(accuracy_score(y_true, y_pred), 2)
        # msg = '[   With Input] accuracy at Top (next)  {} % :: {}'.format(point, accuracy)
        # LOGGER.info(msg)
        # print(msg)
        # TP = 0
        # FN = 0
        # FP = 0
        # # Labels are +1 and -1
        # for i, j in zip(y_true, y_pred):
        #     if i == 1 and i == j:
        #         TP += 1
        #     if i == -1 and j == 1:
        #         FP += 1
        #     if i== 1 and j == -1:
        #         FN += 1
        #
        # precision = round(TP / TP_FN,4)
        # msg = '[   With Input] precision at Top (next)  {} % :: {}'.format(point, precision)
        # LOGGER.info(msg)
        # print(msg)
        # --------------------
        # If we consider all the records till this point as anomalies
        # df_eval2 is sorted by score
        # --------------------
        # df_tmp = df_eval2.head(_count)
        # y_true = list(df_tmp[true_label_name])
        # y_pred = [1] * len(df_tmp)
        # accuracy = round(accuracy_score(y_true, y_pred),2)
        # msg = '[Without Input] accuracy at Top (next)  {} % :: {}'.format(point, accuracy)
        # LOGGER.info(msg)
        # print(msg)
        # TP = 0
        # TP_FN = 0
        # for i, j in zip(y_true, y_pred):
        #     if i == 1 and i == j:
        #         TP += 1
        #     if i == 1: TP_FN += 1
        # precision = round(TP / TP_FN , 2)
        # msg = '[Without Input] precision at Top (next)  {} % :: {}'.format(point, precision)
        # LOGGER.info(msg)
        # print(msg)
    LOGGER.info(results_print.to_string())

# -----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)
parser.add_argument(
    '--classifier_type', choices=['SVM', 'RF'],
    default='RF'
)
args = parser.parse_args()
DIR = args.DIR
classifier_type = args.classifier_type

# -----------------------------------------

setup()
LOGGER = get_logger()
train_df = get_training_data(DIR)
FLAG_network_setup_needed =  not os.path.exists(KNN_dir)
target_df = read_target_data()
record_2_serial_ID_df = get_record_2_serial_ID_df(target_df)
max_K = 100

if FLAG_network_setup_needed:
    network_initialize( )
    process_target_data(
        target_df,
        record_2_serial_ID_df,
        max_K
    )

set_checkpoints = [ 10,20,30,40,50 ]
for checkpoint in set_checkpoints:
    execute_iterative_classification(
        target_df,
        cur_checkpoint = checkpoint
    )

close_logger(LOGGER)
