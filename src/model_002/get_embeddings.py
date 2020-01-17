#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import sys
import pandas as pd
import numpy as np
import sklearn
import glob
import pickle
import random
from joblib import Parallel, delayed
import yaml
import math
from collections import Counter
from collections import OrderedDict

sys.path.append('.')
sys.path.append('./..')

try:
    from . import base_embedding
except:
    import base_embedding
from sklearn.metrics.pairwise import cosine_similarity

# ==================== Global variables ===================== #
CONFIG_FILE = 'config_1.yaml'


# ============================================================ #

def create_coocc_matrix(df, col_1, col_2):
    set_elements_1 = set(list(df[col_1]))
    set_elements_2 = set(list(df[col_2]))
    count_1 = len(set_elements_1)
    count_2 = len(set_elements_2)
    coocc = np.zeros([count_1, count_2])
    df = df[[col_1, col_2]]
    new_df = df.groupby([col_1, col_2]).size().reset_index(name='count')

    for _, row in new_df.iterrows():
        i = row[col_1]
        j = row[col_2]
        coocc[i][j] = row['count']

    print('Col 1 & 2', col_1, col_2, coocc.shape, '>>', (count_1, count_2))
    return coocc


def get_coOccMatrix_dict(df, id_col):
    columns = list(df.columns)
    columns.remove(id_col)
    columns = list(sorted(columns))
    columnWise_coOccMatrix_dict = {}

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_1 = columns[i]
            col_2 = columns[j]
            key = col_1 + '_+_' + col_2
            res = create_coocc_matrix(df, col_1, col_2)
            columnWise_coOccMatrix_dict[key] = res
    columnWise_coOccMatrix_dict = OrderedDict(columnWise_coOccMatrix_dict)
    return columnWise_coOccMatrix_dict



def get_initial_entity_embeddings(
        train_data_file,
        model_data_save_dir,
        DATA_DIR,
        embedding_dims,
        num_epochs,
        id_col='PanjivaRecordID'
):
    train_df = pd.read_csv(train_data_file)

    feature_cols = sorted(list(train_df.columns))
    feature_cols = list(feature_cols)
    feature_cols.remove(id_col)
    domains = feature_cols

    data = train_df[feature_cols].values

    coOcc_dict_file = os.path.join(model_data_save_dir, "coOccMatrix_dict.pkl")
    X_ij_file = os.path.join(model_data_save_dir,"X_ij.pkl")
    domain_dims_file = os.path.join(DATA_DIR,"domain_dims.pkl")

    if os.path.exists(coOcc_dict_file):
        with open(coOcc_dict_file, 'rb') as fh:
            coOccMatrix_dict = pickle.load(fh)
    else:
        coOccMatrix_dict = get_coOccMatrix_dict(train_df, id_col='PanjivaRecordID')
        with open(coOcc_dict_file, "wb") as fh:
            pickle.dump(coOccMatrix_dict, fh, pickle.HIGHEST_PROTOCOL)

    # ----------------
    # Ensure X_ij is in a flattened format ; i < j
    # ----------------
    if os.path.exists(X_ij_file):
        with open(X_ij_file,'rb') as fh:
            X_ij = pickle.load(fh)
    else:
        nd = len(feature_cols)
        num_c = nd * (nd - 1) // 2
        X_ij = np.zeros([data.shape[0], num_c])
        k = 0
        for i in range(len(feature_cols)):
            for j in range(i + 1, len(feature_cols)):
                key = feature_cols[i] + '_+_' + feature_cols[j]
                for d in range(data.shape[0]):
                    e1 = data[d][i]
                    e2 = data[d][j]
                    X_ij[d][k] = coOccMatrix_dict[key][e1][e2]
                k += 1

        with open(X_ij_file, "wb") as fh:
            pickle.dump(X_ij, fh, pickle.HIGHEST_PROTOCOL)

    with open(domain_dims_file, 'rb') as fh:
        domain_dims = pickle.load(fh)

    # X_ij_max needed for scaling
    X_ij_max = []
    for k, v in coOccMatrix_dict.items():
        X_ij_max.append(np.max(v))

    num_domains = len(domain_dims)

    model = base_embedding.get_model(
        domain_dimesnsions=list(domain_dims.values()),
        num_domains = num_domains,
        embed_dim = embedding_dims,
        _X_ij_max=X_ij_max
    )

    base_embedding.train_model(
        model,
        data,
        X_ij,
        file_save_loc=model_data_save_dir,
        epochs=num_epochs
    )

    # ----
    # Save the embeddings (weights) in a dictionary
    # ----
    emb_w = {}
    for i in range(len(feature_cols)):
        dom = feature_cols[i]
        f_path = os.path.join(model_data_save_dir , 'embedding_w_{}.npy'.format(i))
        w = np.load(f_path)
        emb_w[dom] = w

    # ================== 
    # Following GloVe
    # emb ( entity = E in D)
    #  x = 0
    #  For d in {Doamian} - D
    #     x += Sum (CoOcc( E, E_d`)/max(CoOcc( E, E_d`)) *  emb ( entity = E ))
    #  x = 1/2(emb_old(E) + x)
    # ==================

    new_embeddings = {}
    for domain_i in domains:
        new_embeddings[domain_i] = np.zeros(
            emb_w[domain_i].shape
        )

        domain_dim = domain_dims[domain_i]
        # For each entity in domain i 
        for entity_id in range(domain_dim):
            res = 0
            # For each entity in domain j != i
            for domain_j in domains:
                if domain_j == domain_i: continue
                pair = sorted([domain_i, domain_j])

                key = '_+_'.join(pair)
                coOcc_matrix = coOccMatrix_dict[key]
                if domain_i == pair[0]:
                    arr = coOcc_matrix[entity_id, :]
                else:
                    arr = coOcc_matrix[:, entity_id]

                sum_co_occ = max(np.sum(arr), 1)
                scale = np.reshape(arr / sum_co_occ, [-1, 1])

                emb_domain_j = emb_w[domain_j]
                res_j = np.sum(scale * emb_domain_j, axis=0)
                res = res + res_j

            res = 0.5 * (res + emb_w[domain_i][entity_id])
            new_embeddings[domain_i][entity_id] = res

    # Write the embeddings to file 
    for domain_i in domains:
        print(' >> ', domain_i)
        file_name = os.path.join(
            model_data_save_dir,
            'init_embedding' + domain_i + '.npy'
        )
        np.save(
            file=file_name,
            arr=new_embeddings[domain_i]
        )

    # =================================
    # This is only for testing whether the model works
    # Usually not called, only for debugging
    # =================================
    def test():
        hscode = 25
        # find the 10 closest  to ShipmentDestination to HSCode in data
        df = train_df.loc[train_df['HSCode'] == hscode]
        df = df.groupby(['HSCode', 'ShipmentDestination']).size().reset_index(name='counts')
        df = df.sort_values(by=['counts'])

        k_closest = df.tail(10)['ShipmentDestination'].values
        print(k_closest)

        # hs_code_vec = wt[0][hscode] + bias[0][hscode]
        hs_code_vec = new_embeddings['HSCode'][hscode]

        shp_dest_vec = []
        wt = new_embeddings['ShipmentDestination']
        for i in range(wt.shape[0]):
            r = wt[i]
            shp_dest_vec.append(r)

        res = {}
        for i in range(wt.shape[0]):
            a = np.reshape(shp_dest_vec[i], [1, -1])
            b = np.reshape(hs_code_vec, [1, -1])
            res[i] = cosine_similarity(a, b)

        new_df = pd.DataFrame(list(res.items()))
        new_df = new_df.sort_values(by=[1])
        print(new_df.tail(10))

    return new_embeddings


# ======================================================== #


