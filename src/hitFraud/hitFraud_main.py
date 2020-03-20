import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
from scipy.sparse import csr_matrix
from src.utils import coOccMatrixGenerator
from scipy import sparse
import argparse
import pickle
from pandarallel import pandarallel
pandarallel.initialize()
try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher
domain_dims = None
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
# ------------------------------------ #

# Input :
# Stage 1 :
# Training data to create network
# Metapaths
# Stage 2
# DataFrame of [ Test Transactions ids, Scores , Entities ]
# ------------------------------------- #

def get_domain_dims(DIR):
    global domain_dims
    with open(
            os.path.join(
                './../../generated_data_v1/',
                DIR,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)
    return


def get_training_data(DIR):
    SOURCE_DATA_DIR = './../../generated_data_v1'
    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR, DIR)
    return data


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


def get_transition_matrix(domain1, domain2):
    global coOccDict
    if domain1 < domain2:
        key = domain1 + '_+_' + domain2
        return coOccDict[key]
    else:
        key = domain2 + '_+_' + domain1
        return np.transpose(coOccDict[key])


class MP_object:
    id = 0
    @staticmethod
    def assign_id():
        t =  MP_object.id
        MP_object.id = MP_object.id + 1
        return t

    def __init__(self, MP_list):
        global domain_dims
        # symmetric
        self.mp = MP_list + MP_list[::-1][1:]
        self.id = MP_object.assign_id()

        arr_list = []
        for r1,r2 in zip(self.mp[:-1],self.mp[1:]):
            mat = get_transition_matrix(r1, r2)
            mat = csr_matrix(mat)
            inplace_csr_row_normalize_l1(mat)
            arr_list.append(mat)

        domain_sizes = [ domain_dims[d]  for d in self.mp ]

        # Calculate z
        # z = D * P * y
        # z is [n,1]
        # y is [n,1]
        # D is [n,k]
        # P is [k,n]
        # P is the multiplication of all the matrices

        # optimize by breaking the multiplication into 2 parts
        mult_order = optimal_parenthesization(domain_sizes)


        def mult(_list_indices):
            if type(_list_indices[0]) == int  and type(_list_indices[1])==int:
                mat_a = arr_list[_list_indices[0]]
                mat_b = arr_list[_list_indices[1]]
                return mat_a * mat_b
            elif type(_list_indices[0]) == int and type(_list_indices[1])==list:
                mat_a = arr_list[_list_indices[0]]
                mat_b = mult( _list_indices[1])
                return mat_a * mat_b
            elif type(_list_indices[0]) == list and type(_list_indices[1]) == int:
                mat_a = mult(_list_indices[0])
                mat_b = arr_list[_list_indices[1]]
                return mat_a * mat_b
            else:

                mat_a = mult(_list_indices[0])
                mat_b = mult(_list_indices[1])
                return mat_a * mat_b

        self.P = mult(mult_order)
        self.D = sparse.spdiags(
            data=np.reciprocal(
                np.reshape(
                    np.sum(self.P, axis=1),
                    [-1])
            ),
            diags=0,
            m=self.P.shape[0],
            n=self.P.shape[0]
        )

    def calc_z(self, y):
        n = y.shape[0]
        res =  self.D * self.P
        res =  res * y
        return res

# ------------------------------------- #
# Calculate the initial network
# -------------------------------------- #
def network_creation(
        train_x_df,
        MP_list,
        id_col='PanjivaRecordID'
):
    global coOccDict
    coOccDict_file = 'model_use_data/coOccDict.pkl'
    if os.path.exists(coOccDict_file):
        with open(coOccDict_file,'rb') as fh :
            coOccDict = pickle.load(fh)
    else:
        coOccDict = coOccMatrixGenerator.get_coOccMatrix_dict(train_x_df, id_col)
        with open(coOccDict_file, 'wb') as fh:
            pickle.dump(
                coOccDict,
                fh,
                pickle.HIGHEST_PROTOCOL
            )
    for mp in MP_list :
        mp_obj = MP_object(mp)


DIR ='us_import1'
get_domain_dims(DIR)
train_x = get_training_data(DIR)
MP_list = get_metapath_list()

network_creation(
    train_x,
    MP_list
)

