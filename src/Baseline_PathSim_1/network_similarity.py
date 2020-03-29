#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
from scipy import sparse
from scipy.sparse import load_npz
from scipy.sparse import save_npz
import argparse
import pickle
from hashlib import md5
from scipy.sparse import csr_matrix
import pickle
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import multiprocessing
import joblib
from joblib import Parallel,delayed

sys.path.append('./../..')
sys.path.append('./..')
from joblib import Parallel, delayed
from src.utils import coOccMatrixGenerator

try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher

nodeObj_dict_file = 'nodeObj_dict.pkl'
REFRESH_NODES = False
id_col = 'PanjivaRecordID'
nodeObj_Dict = None
model_use_data_DIR = None
domain_dims = None

# ---------------------------------------

# ------------------------------------------------------------3
# Call this to set up global variables
# ------------------------------------------------------------
def initialize(_dir, _model_use_data_DIR = None):
    global DIR
    global model_use_data_DIR
    global domain_dims

    DIR = _dir
    domain_dims = get_domain_dims(DIR)

    if _model_use_data_DIR is None:
        model_use_data_DIR = 'model_use_data'
    else:
        model_use_data_DIR = _model_use_data_DIR

    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    if not os.path.exists(os.path.join(model_use_data_DIR, DIR)):
        os.mkdir(os.path.join(model_use_data_DIR, DIR))
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    return


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

def read_target_data(
        DATA_SOURCE, DIR
):
    csv_f_name = 'scored_test_data.csv'
    df = pd.read_csv(
        os.path.join(
            DATA_SOURCE,
            DIR,
            csv_f_name), index_col=None
    )
    return df

# -----------------------------------------------
def matrix_multiply(
        matrix_list,
        _csr = True

):
    list_dimensions = [ matrix_list[0].shape[0], matrix_list[0].shape[1]]
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



    def aux_mul(a,b):
        if _csr:
            return a *b
        else:
            return np.matmul(a,b)

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
    return  mult(mult_order)

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
#-----------------------------------------------------------------------

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
        saved_file_name = 'mp_object_' + signature + '.pkl'
        saved_file_path = os.path.join(
            model_use_data_DIR,
            saved_file_name
        )

        if os.path.exists(saved_file_path):
            print(signature)
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
        f_name = 'sim_matrix_mp_' + str(self.id) + '.npz'

        t_df = t_df.sort_values(by=[id_col],ascending=True)
        n = len(t_df)
        simMatrix_path = os.path.join(
            model_use_data_DIR,
            f_name
        )
        if os.path.exists(simMatrix_path):
            simMatrix = load_npz(simMatrix_path)

        else:
            print(domain_dims)
            conn_domain = self.mp[0]
            print(conn_domain)
            A_t_d = np.zeros([n, domain_dims[conn_domain]])
            d_vals = list(t_df[conn_domain])
            A_t_d[np.arange(n), d_vals] = 1
            A_t_d = csr_matrix(A_t_d)
            simMatrix = A_t_d * (self.CM * A_t_d.transpose())
            # simMatrix = simMatrix.toarray()
            save_npz(
                simMatrix_path,
                simMatrix
            )
            D = simMatrix.diagonal()
            for i in range(n):
                for j in range(i):
                    simMatrix[i][j] = 2 * simMatrix[i][j]/ (D[i] + D[j])
                    simMatrix[j][i] = simMatrix[i][j]
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

DIR = 'us_import1'
initialize(DIR)
df = get_training_data(DIR)

print(domain_dims)
MP_list = get_metapath_list()
list_mp_obj = network_creation(
    df,MP_list
)

target_df = read_target_data(
    DATA_SOURCE='./../../AD_system_output',
    DIR = DIR
)
target_df = target_df.head(1000)

def aux_set_PS ( mp_obj , target_df, domain_dims):
    mp_obj.calc_PathSim(target_df, domain_dims)
    return

for mp_obj in list_mp_obj:
    mp_obj.calc_PathSim(
        target_df,
        domain_dims
    )





