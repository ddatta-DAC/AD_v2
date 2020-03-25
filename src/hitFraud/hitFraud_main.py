import math
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
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

    def calc_z(
            self,
            data
    ):
        y = np.reshape(list(data['y']), [-1, 1])
        n = y.shape[0]
        starting_domain = self.mp[0]
        d = starting_domain
        A_t_d = np.zeros([n, domain_dims[d]])
        d_vals = list(data[d])
        A_t_d[np.arange(n), d_vals] = 1
        A_t_d = csr_matrix(A_t_d)
        _P = self.P

        # D = sparse.spdiags(
        #     data=np.reciprocal(
        #         np.reshape(
        #             np.sum(_P, axis=1),
        #             [-1])
        #     ),
        #     diags=0,
        #     m=_P[0],
        #     n=_P[0]
        # )

        res = A_t_d * (_P * (A_t_d.transpose() * csr_matrix(y)))
        res = res.toarray()
        res = np.reshape(res, -1)

        print('Res ::', res.shape)
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
    csv_f_name = 'scored_test_data.csv'
    df = pd.read_csv(
        os.path.join(
            DATA_SOURCE,
            DIR,
            csv_f_name), index_col=None
    )
    return df


# --------------------------------------
# Assign "human_labels"
# checkpoints are 10%, 20%, 30%, 40%, 50%
# -----------------------------------------------
def exec_classifier(
        list_mp_obj,
        checkpoint=10,
        classifier_type=None
):
    global DIR

    cur_checkpoint = 10
    label_col = 'y'
    id_col = 'PanjivaRecordID'
    df = read_target_data(
        DATA_SOURCE='./../../AD_system_output',
        DIR=DIR
    )
    df_master = df.copy()

    record_count = len(df)

    # count of how many labelled and unlabelled datapoints
    l_count = int(record_count * cur_checkpoint / 100)
    u_count = record_count - l_count

    # The records are sorted by score
    df_L = df.head(l_count).copy()
    df_U = df.tail(u_count).copy()
    labelled_instance_ids = list(df_L[id_col])
    unlabelled_instance_ids = list(df_U[id_col])

    def set_y(row):
        if row['fraud'] == True:
            return 1  # labelled True
        elif row['fraud'] == False:
            return -1  # labelled false
        else:
            return 0  # unknown

    # Set labels for instances which have been "Labelled" by humans
    df_L[label_col] = 0
    df_U[label_col] = 0
    df_L[label_col] = df_L.parallel_apply(set_y, axis=1)

    df_known_y_values = df_L[[id_col, label_col]]

    # df is the working dataframe
    df_UL = df_L.append(df_U, ignore_index=True)
    del df_UL['fraud']
    del df_UL['anomaly']

    # ---------------------------------------------------
    # For each metapath calculate the meta path feature z
    # ---------------------------------------------------
    z_list = []
    for mp_obj in list_mp_obj:
        z = mp_obj.calc_z(df_UL)
        _id = mp_obj.id
        zcol = 'z' + str(_id)
        z_list.append(zcol)
        df_UL[zcol] = list(z)

    one_hot_columns = list(domain_dims.keys())
    clf_train_df = pd.get_dummies(
        df_UL,
        columns=one_hot_columns
    )

    # Train initial Classifier model
    if classifier_type == 'RF':
        clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            verbose=1
        )
    elif classifier_type == 'SVM':
        clf = SVC(
            kernel='poly',
            degree='4'
        )

    # ----------------------------------------------------
    # Train initial model only on the labelled data
    # Input features should be  [ entities, score , {z} ]
    # ----------------------------------------------------

    clf_train_df_L = clf_train_df.loc[
        clf_train_df[id_col].isin(labelled_instance_ids)
    ]
    clf_train_df_U = clf_train_df.loc[
        clf_train_df[id_col].isin(unlabelled_instance_ids)
    ]
    y = list(clf_train_df_L[label_col])

    tmp = clf_train_df_L.copy()
    remove_cols = [id_col, label_col]
    for rc in remove_cols:
        del tmp[rc]

    x = tmp.values
    clf.fit(x, y)

    # -----------------------------------------
    # Bootstrap ::  obtain initial labels of U using the trained classifier
    # Set z = 0
    # -----------------------------------------

    tmp = clf_train_df_U.copy()
    for _zcol in z_list:
        tmp[_zcol] = 0

    for rc in remove_cols:
        del tmp[rc]
    x = tmp.values
    pred_y = clf.predict(x)
    pred_y = np.reshape(pred_y, -1)
    clf_train_df_U[label_col] = pred_y

    # Keep a df with entity ids for meta path feature calculations
    df_U_copy = df_U.copy()
    df_U_copy[label_col] = pred_y

    df_iterative = df_L.copy().append(df_U_copy, ignore_index=True)

    try:
        del df_iterative['fraud']
        del df_iterative['anomaly']
    except:
        pass

    '''
    Algorithm outline ::
    Repeat : Till Convergence or iter > max_iter
    Following ICA ( See Charu Aggarwal Text Pg 325, Ch 10 :  
        at each step include the most confident labels
    For abs(label_value) >  epsilon  ; 
        set new_label = sign(label_val)
        epsilon = 50th percentile of absolute label values
    Clamp down known labels

    '''

    def restore_known_labels(row, ref_df):
        r = ref_df.loc[ref_df[id_col] == row[id_col]]
        if len(r) > 0:
            return list(r[label_col])[0]
        else:
            return row[label_col]

    max_iter = 10
    iter = 1
    epsilon = 0.5

    def set_label_by_sign(row, epsilon):
        if abs(row[label_col]) > epsilon:
            return np.sign(row[label_col])
        else:
            return 0

    # ----------------------------------

    while True:
        print(' Iteration :', iter)
        # Recalculate z
        for mp_obj in list_mp_obj:
            z = mp_obj.calc_z(df_iterative)
            _id = mp_obj.id
            zcol = 'z' + str(_id)
            df_iterative[zcol] = list(z)

        # Retrain classifier
        tmp_df = df_iterative.copy()

        for rc in remove_cols:
            del tmp_df[rc]

        tmp_df = pd.get_dummies(
            tmp_df,
            columns=one_hot_columns
        )
        x = tmp_df.values
        pred_y = clf.predict(x)
        pred_y = np.reshape(pred_y, [-1])
        df_iterative['y'] = pred_y

        # Clamp down the known labels
        df_iterative.loc[:, label_col] = df_iterative.parallel_apply(
            restore_known_labels,
            axis=1,
            args=(df_known_y_values,)
        )
        df_iter_L = df_iterative.loc[df_iterative[id_col].isin(labelled_instance_ids)]
        # Select the most confident labels
        df_iter_U = df_iterative.loc[df_iterative[id_col].isin(unlabelled_instance_ids)]
        # --------------------------
        # set labels as per sign
        # --------------------------
        df_iter_U.loc[:, label_col] = df_iter_U.parallel_apply(
            set_label_by_sign, axis=1, args=(epsilon,)
        )
        epsilon = max(math.pow(epsilon, 0.75), 0.05)
        df_iterative = df_iter_L.append(df_iter_U, ignore_index=True)

        # check if any unlabelled nodes
        if (iter > max_iter) and (0 not in list(df_iterative[label_col])):
            break
        iter += 1

    df_eval = df_iterative.loc[df_iterative[id_col].isin(unlabelled_instance_ids)]

    # Add in actual label columns
    def place_true_labels(row, ref_df):
        _id = row[id_col]
        r = list(df.loc[ref_df[id_col]==_id]['fraud'])[0]
        if r is True:
            return 1
        else:
            return -1
    true_label_name = 'y_true'
    df_eval[true_label_name] = df_eval.parallel_apply(
        place_true_labels,
        axis=1,
        args=(df_master,)
    )

    # We are trying to understand how input so far will improve the output in next stage
    # So we try to see the metrics in the next 10% of the data
    from sklearn.metrics import precision_score
    df_eval = df_eval.sort_values(by=['score'],ascending=True)
    # Take next 20% of data
    for point in [10,20,30,40,50]:
        _count = int(len(df_master)*point/100)
        df_tmp = df_eval.head(_count)
        y_true = list(df_tmp[true_label_name])
        y_pred = list(df_tmp[label_col])
        precision = precision_score(y_true, y_pred)
        print('Precision at top {} % :: {}'.format(point, precision))

    return






# -------------------------------------------------------------------------------
# Set up parameters
# -------------------------------------------------------------------------------

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

exec_classifier(
    list_mp_obj,
    checkpoint= 10,
    classifier_type = classifier_type
)