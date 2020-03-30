# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
import sys
from numpy import load as load_np
from numpy import save as save_np
from scipy.linalg.blas import sgemm
from hashlib import md5
from scipy.sparse import csr_matrix
import pickle
import sys
import os
import pandas as pd
import numpy as np
import multiprocessing
sys.path.append('.')
sys.path.append('./..')
sys.path.append('./../..')
from joblib import Parallel, delayed
import pickle
import argparse
import multiprocessing
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
TARGET_DATA_SOURCE = './../../AD_system_output'
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

# ------------------------------------------------------------
# ---------         First function to be executed
# Call this to set up global variables
# ------------------------------------------------------------

def setup():
    global DIR
    global config_file
    global model_use_data_DIR
    global TARGET_DATA_SOURCE
    global domain_dims
    global KNN_k
    global data_max_size
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


# ---------------------------------------
def get_training_data(DIR):
    SOURCE_DATA_DIR = './../../generated_data_v1'
    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR, DIR)
    return data


def get_domain_dims(DIR):
    with open(
            os.path.join(
                './../../generated_data_v1',
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
        if os.path.exists(simMatrix_path):
            simMatrix = load_np(simMatrix_path)
            print(simMatrix.shape)

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

            save_np(
                simMatrix_path,
                simMatrix
            )

        self.simMatrix = simMatrix
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
        _id = mp_obj.id
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

def process_target_data(
        target_df,
        _record_2_serial_ID_df,
        K=100
):
    global record_2_serial_ID_df
    global list_MP_OBJ

    record_2_serial_ID_df = _record_2_serial_ID_df
    args = [(_obj , target_df.copy(), domain_dims.copy()) for _obj in list_MP_OBJ]
    n_jobs = multiprocessing.cpu_count()
    with Pool(n_jobs) as p:
        res = p.map(aux_precompute_PathSimCalc ,args)

    # for mp_obj in list_MP_OBJ:
    #     mp_obj.calc_PathSim(
    #         target_df,
    #         domain_dims
    #     )

    save_Dir = 'KNN'
    save_Dir = os.path.join(model_use_data_DIR, save_Dir)
    if not os.path.exists(save_Dir):
        os.mkdir(save_Dir)

    n_jobs = multiprocessing.cpu_count()
    args = [(_record_ID, K,save_Dir) for _record_ID in list(target_df[id_col])]
    with Pool(n_jobs) as p:
        res = p.map(set_up_closest_K_by_RecordID ,args)

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

    csv_f_name = 'scored_test_data.csv'
    df = pd.read_csv(
        os.path.join(
            TARGET_DATA_SOURCE,
            DIR,
            csv_f_name), index_col=None
    )
    df = df.sample(data_max_size)
    df = df.sort_values(
        by=[id_col]
    )
    return df


def get_record_2_serial_ID_df(target_df):
    global id_col
    global model_use_data_DIR
    record_2_serial_file = os.path.join(model_use_data_DIR, 'record_2_serial_ID.csv')

    record_2_serial_ID = {
        e[1]: e[0] for e in enumerate(list(target_df[id_col]), 0)
    }
    record_2_serial_ID_df = pd.DataFrame(
        record_2_serial_ID.items(), columns=[id_col, 'Serial_ID']
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
train_df = get_training_data(DIR)
network_initialize( )

target_df = read_target_data()

record_2_serial_ID_df = get_record_2_serial_ID_df(target_df)
process_target_data(
    target_df,
    record_2_serial_ID_df,
    KNN_k
)
