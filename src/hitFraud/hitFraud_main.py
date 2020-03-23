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

from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
from hashlib import md5

domain_dims = None
MODEL_DATA_DIR = None


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
        global MODEL_DATA_DIR
        signature = MP_object.get_signature(MP)
        saved_file_name = 'mp_object_' + signature + '.pkl'
        saved_file_path = os.path.join(MODEL_DATA_DIR, saved_file_name)

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

        # symmetric
        self.mp = MP + MP[::-1][1:]
        self.id = MP_object.assign_id()

        arr_list = []
        for r1, r2 in zip(self.mp[:-1], self.mp[1:]):
            mat = get_transition_matrix(r1, r2)
            mat = csr_matrix(mat)
            inplace_csr_row_normalize_l1(mat)
            arr_list.append(mat)

        domain_sizes = [domain_dims[d] for d in self.mp]
        # P is the multiplication of all the matrices

        # optimize by breaking the multiplication into 2 parts
        mult_order = optimal_parenthesization(domain_sizes)

        def mult(_list_indices):
            if type(_list_indices[0]) == int and type(_list_indices[1]) == int:
                mat_a = arr_list[_list_indices[0]]
                mat_b = arr_list[_list_indices[1]]
                return mat_a * mat_b
            elif type(_list_indices[0]) == int and type(_list_indices[1]) == list:
                mat_a = arr_list[_list_indices[0]]
                mat_b = mult(_list_indices[1])
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


    # Calculate z
    # z = D * P * y
    # z is [n,1]
    # y is [n,1]
    # D is [n,k]
    # P is [k,n]
    def calc_z(self, data):
        y = np.reshape(list(data['y']),[-1,1])
        n = y.shape[0]
        starting_domain = self.mp[0]
        d = starting_domain
        A_t_d = np.zeros(n, domain_dims[d])
        d_vals = list(data[d])
        A_t_d[np.arange(n),d_vals] = 1
        A_t_d = csr_matrix(A_t_d)
        _P = self.P
        _P = A_t_d * _P
        _P = _P * A_t_d.transpose()

        D = sparse.spdiags(
            data=np.reciprocal(
                np.reshape(
                    np.sum(_P, axis=1),
                    [-1])
            ),
            diags=0,
            m=_P[0],
            n=_P[0]
        )

        res = D * _P
        res = res * y
        res = res.todense()
        res = np.reshape(res,[-1])
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


# --------------------------------------

def read_target_data(
        DATA_SOURCE, DIR
):
    csv_f_name = 'scored_test_dat.csv'
    df = pd.read_csv(
        os.path.join(
            DATA_SOURCE,
            DIR,
            csv_f_name), index_col=None
    )
    return df


# --------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)
args = parser.parse_args()
DIR = args.DIR
# --------------------------------------

MODEL_DATA_DIR = os.path.join('model_use_data', DIR)
if not os.path.exists(MODEL_DATA_DIR):
    os.mkdir(MODEL_DATA_DIR)

get_domain_dims(DIR)
train_x = get_training_data(DIR)
MP_list = get_metapath_list()

list_mp_obj = network_creation(
    train_x,
    MP_list
)

# Assign "human_labels"
# checkpoints are 10%, 20%, 30%, 40%, 50%
df = read_target_data(
    DATA_SOURCE='./../../AD_system_output',
    DIR=None
)
record_count = len(df)
cur_checkpoint = 10
l_count = int(record_count * cur_checkpoint / 100)
u_count = record_count - l_count
labelled_df1 = df.head(l_count).copy()
labelled_df2 = df.tail(u_count).copy()


def set_y(row):
    if row['fraud'] == True:
        return 1  # labelled True
    elif row['fraud'] == False:
        return -1  # labelled false
    else:
        return 0  # unknown

id_col = 'PanjivaRecordID'

labelled_df1['y'] = 0
labelled_df2['y'] = 0

# Set labels for instances which have been "Labelled" by humans
labelled_df1['y'] = labelled_df1.parallel_apply(set_y, axis=1)

df = labelled_df1.append(labelled_df2,ignore_index=True)

classif_features = []
# calculate meta path features based on each of the metapths
labelled_instance_ids = list(labelled_df1[id_col])
unlabelled_instance_ids = list(labelled_df1[id_col])
for mp_obj in list_mp_obj:
    z = mp_obj.calc_z(df)
    _id = mp_obj.id
    zcol = z+str(_id)
    df[zcol] = list(z)
    classif_features.append(zcol)

# Train model
# Input features should be the entities and {z}
one_hot_columns = list(domain_dims.keys())
df = pd.get_dummies(
    df,
    columns = one_hot_columns
)
df_train = df.loc[df[id_col].isin(labelled_instance_ids)]
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    verbose=1
)

tmp = df_train.copy()
del tmp[id_col]
y = list(tmp['y'])
del tmp[y]
x = tmp.values
clf.fit(X = x, y = y)


# Train initial model only on the labelled data
