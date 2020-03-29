#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

import pickle
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import multiprocessing

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


# ---------------------------------------

# ------------------------------------------------------------3
# Call this to set up global variables
# ------------------------------------------------------------
def initialize(_dir, _model_use_data_DIR):
    global DIR
    global model_use_data_DIR

    DIR = _dir
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

    mult_order = optimal_parenthesization(list_dimensions)

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

    return  mult(mult_order)

# -------------------------------------------------------------------

class MP_object:
    id = 0
    @staticmethod
    def assign_id():
        t = MP_object.id
        MP_object.id = MP_object.id + 1
        return t

    @staticmethod
    def get_signature(MP_list):
        _signature = ''.join(sorted([''.join(_) for _ in MP_list]))
        signature = str(md5(str.encode(_signature)).hexdigest())
        return signature

    @staticmethod
    def GET_mp_obj(MP):
        global model_use_data_DIR
        signature = MP_object.get_signature(MP)
        saved_file_name = 'mp_object_' + signature + '.pkl'
        saved_file_path = os.path.join(model_use_data_DIR, saved_file_name)

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

        # set up the commuting matrix for PathSim



# -------------------------------------------------------------------
def network_creation(
        train_x_df,
        MP_list,
        id_col='PanjivaRecordID'
):
    global coOccDict
    global MODEL_DATA_DIR

    coOccDict_file = os.path.join(MODEL_DATA_DIR, 'coOccDict.pkl')
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

