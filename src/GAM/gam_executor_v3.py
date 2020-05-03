#!/usr/bin/env python
# coding: utf-8

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import copy
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
sys.path.append('./../..')
sys.path.append('./..')
from time import time
from pandarallel import pandarallel

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score

pandarallel.initialize()

import argparse
from datetime import datetime
import multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler, SequentialSampler
import yaml
import pickle
import torch
import torch.nn as nn
import logging
from torch import FloatTensor as FT
from torch import LongTensor as LT
from torch.nn import functional as F

DEVICE = None

try:
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    DEVICE = torch.device(dev)
    print('Set Device :: ', DEVICE)
    print('Cuda available ::', torch.cuda.is_available(),
          'Cuda current device ::', torch.cuda.current_device(),
          torch.cuda.get_device_name(0))
except:
    print('No CUDA')

try:
    from torch import has_cudnn

    if has_cudnn:
        torch.cudnn.benchmark = False
        print('Set cudnn benchmark to True')
except:
    pass

try:
    from .gam_module import agreement_net_v2 as gam_net
    from .gam_module import gam_loss
    from .clf_net import clf_net_v2 as clf_MLP
    from .clf_net import clf_loss as clf_loss
    from .record_node import graph_net_v2 as graph_net
    from .torch_data_loader import type1_Dataset
    from .torch_data_loader import dataGeneratorWrapper
    from . import train_utils
    from . import data_preprocess
    from .torch_data_loader import pairDataGenerator_v1
    from .torch_data_loader import singleDataGenerator
    from .torch_data_loader import  balanced_pair_Generator_v2
    from .torch_data_loader import pairDataGenerator_v2
    from .src.Classifiers import wide_n_deep_model as clf_WIDE_N_DEEP
    from .src.Classifiers import deepFM  as clf_DEEP_FM
    from .GAM_SS_module import SS_network
except:
    from gam_module import agreement_net_v2 as gam_net
    from gam_module import gam_loss
    from clf_net import clf_net_v2 as clf_MLP
    from clf_net import clf_loss as clf_loss
    from record_node import graph_net_v2 as graph_net
    from torch_data_loader import type1_Dataset
    from torch_data_loader import pairDataGenerator_v1
    from torch_data_loader import pairDataGenerator_v2
    from torch_data_loader import singleDataGenerator
    from src.Classifiers import wide_n_deep_model as clf_WIDE_N_DEEP
    from src.Classifiers import deepFM  as clf_DEEP_FM
    from torch_data_loader import singleDataGenerator
    import data_preprocess
    import train_utils
    from torch_data_loader import balanced_pair_Generator_v2
    from GAM_SS_module import SS_network

# ==================================== #

config_file = 'config.yaml'
CONFIG = None
DATA_SOURCE_DIR_1 = None
DATA_SOURCE_DIR_2 = None
model_use_data_DIR = None
DIR = None
logger = None
Logging_Dir = None
domain_dims = None
score_col = 'score'
fraud_col = 'fraud'
anomaly_col = 'anomaly'
id_col = 'PanjivaRecordID'
label_col = 'y'
true_label_col = 'y_true'
node_emb_dim = 128
feature_col_list = []
serial_mapping_df = None
is_labelled_col = 'labelled'
matrix_node_emb_path = None
confidence_bound = 0.2
epochs_f = 0
epochs_g = 0
log_interval_f = 10
log_interval_g = 10
max_IC_iter = 5
clf_mlp_layer_dimesnions = []
gam_encoder_dimensions_mlp = []
batch_size_g = 128
batch_size_f = 128
batch_size_r = 128
F_classifier_type = None
WnD_dnn_layer_dimensions = None
deepFM_dnn_layer_dimensions = None
LOGGER = None

def setup_config(_DIR):
    global CONFIG
    global config_file
    global DATA_SOURCE_DIR_1
    global DATA_SOURCE_DIR_2
    global DIR
    global Logging_Dir
    global model_use_data_DIR
    global domain_dims
    global feature_col_list
    global serial_mapping_df
    global serialized_feature_col_list
    global matrix_node_emb_path
    global confidence_bound
    global epochs_f
    global epochs_g
    global log_interval_f
    global log_interval_g
    global max_IC_iter
    global clf_mlp_layer_dimesnions
    global gam_encoder_dimensions_mlp
    global batch_size_g
    global batch_size_f
    global batch_size_r
    global F_classifier_type
    global WnD_dnn_layer_dimensions
    global deepFM_dnn_layer_dimensions
    global LOGGER

    if _DIR is not None:
        DIR = _DIR

    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)
    if F_classifier_type is None:
        F_classifier_type = CONFIG['clf_type']

    DATA_SOURCE_DIR_1 = CONFIG['DATA_SOURCE_DIR_1']
    DATA_SOURCE_DIR_2 = CONFIG['DATA_SOURCE_DIR_2']

    DATA_SOURCE_DIR_1 = os.path.join(DATA_SOURCE_DIR_1, DIR)
    DATA_SOURCE_DIR_2 = os.path.join(DATA_SOURCE_DIR_2, DIR)

    model_use_data_DIR = CONFIG['model_use_data_DIR']
    if not os.path.exists(model_use_data_DIR): os.mkdir(model_use_data_DIR)
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    if not os.path.exists(model_use_data_DIR): os.mkdir(model_use_data_DIR)

    with open(
            os.path.join(
                DATA_SOURCE_DIR_1,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)

    feature_col_list = list(sorted(domain_dims.keys()))
    serialized_feature_col_list = ['_' + _ for _ in feature_col_list]
    serial_mapping_df_path = os.path.join(
        CONFIG['serial_mapping_df_loc'],
        DIR,
        CONFIG['serial_mapping_df_name']
    )

    serial_mapping_df = pd.read_csv(serial_mapping_df_path, index_col=None)
    matrix_node_emb_path = os.path.join(CONFIG['matrix_node_emb_loc'], DIR, CONFIG['matrix_node_emb_file'])
    confidence_bound = CONFIG['confidence_bound']
    epochs_g = CONFIG['epochs_g']
    epochs_f = CONFIG['epochs_f']
    log_interval_f = CONFIG['log_interval_f']
    log_interval_g = CONFIG['log_interval_g']
    max_IC_iter = CONFIG['max_IC_iter']
    clf_mlp_layer_dimesnions = [
        int(_)
        for _ in CONFIG['classifier_mlp_layers_1'].split(',')
    ]
    WnD_dnn_layer_dimensions = [
        int(_)
        for _ in CONFIG['WnD_dnn_layer_dimensions'].split(',')
    ]
    deepFM_dnn_layer_dimensions = [
        int(_)
        for _ in CONFIG['deepFM_dnn_layer_dimensions'].split(',')
    ]
    gam_encoder_dimensions_mlp = [
        int(_)
        for _ in CONFIG['gam_encoder_dimensions_mlp'].split(',')
    ]

    batch_size_g = CONFIG['batch_size_g']
    batch_size_f = CONFIG['batch_size_f']
    batch_size_r = CONFIG['batch_size_r']
    Logging_Dir = CONFIG['Logging_Dir']
    LOGGER = get_logger()
    LOGGER.info(str(datetime.utcnow()))
    return


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
    return logger


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    return


def read_matrix_node_emb(matrix_node_emb_path):
    emb = np.load(matrix_node_emb_path)
    return emb


def regularization_loss(
        g_ij,
        fi_yj
):
    g_ij = g_ij.view(-1)
    val1 = (fi_yj[0] - fi_yj[1]) ** 2
    val2 = val1.float() * g_ij
    val3 = (val2).mean()
    return val3


def train_model(
        NN,
        df,
        normal_data_samples_df,
        features_F,
        features_G
):
    global epochs_f
    global epochs_g
    global log_interval_f
    global log_interval_g
    global max_IC_iter
    global serialized_feature_col_list
    global feature_col_list
    global DEVICE
    global batch_size_g
    global batch_size_f
    global batch_size_r
    global LOGGER

    num_epochs_g = epochs_g
    num_epochs_f = epochs_f
    results_print = None
    # num_proc = multiprocessing.cpu_count()
    lambda_LL = 0.05
    lambda_UL = 0.1
    lambda_UU = 0.01

    df_L = train_utils.extract_labelled_df(df)
    df_U = train_utils.extract_unlabelled_df(df)
    df_L_original = df_L.copy()
    df_L = df_L.copy()

    # df_L, df_L_validation = train_utils.obtain_train_validation(
    #     df_L
    # )

    clf_en1 = None

    # Add in normal data to validation data
    # df_L_validation = df_L_validation.append(
    #     normal_data_samples_df.sample(len(df_L_validation)),
    #     ignore_index=True
    # )

    df_U_original = df_U.copy()
    # print('>> Data set lengths :', len(df_L), len(df_L_validation), len(df_U))

    current_iter_count = 0
    continue_training = True

    while continue_training:
        # GAM gets inputs as embeddings, which are obtained through the graph embeddings
        # that requires serialized feature ids
        # g_feature_cols = serialized_feature_col_list
        NN.train_mode = 'g'
        try:
            params_list_g = [_ for _ in NN.graph_net.parameters()]
            params_list_g = params_list_g + ([_ for _ in NN.agreement_net.parameters()])
        except:
            params_list_g = [_ for _ in NN.module.graph_net.parameters()]
            params_list_g = params_list_g + ([_ for _ in NN.module.agreement_net.parameters()])

        print('# of parameters to be optimized for g ', len(params_list_g))
        optimizer_g = torch.optim.Adam(
            params_list_g,
            lr=0.025
        )
        try:
            params_list_f = [_ for _ in NN.graph_net.parameters()]
            params_list_f = params_list_f + [_ for _ in NN.clf_net.parameters()]
        except:
            params_list_f = [_ for _ in NN.module.graph_net.parameters()]
            params_list_f = params_list_f + [_ for _ in NN.module.clf_net.parameters()]
        print('# of parameters to be optimized for f ', len(params_list_f))
        optimizer_f = torch.optim.Adam(
            params_list_f,
            lr=0.05
        )

        final_epoch_g = False  # To check convergence
        if NN.train_mode == 'g':

            print('Training Agreement model .... ')
            optimizer_g.zero_grad()
            prev_loss = 0
            iter_below_tol = 0

            for epoch in range(num_epochs_g):

                data_G = balanced_pair_Generator_v2(
                    df=df_L,
                    x_col=features_G,
                    y_col=label_col,
                    allow_refresh=False,
                    device=DEVICE
                )
                print('Epoch [g]', epoch)
                record_loss = []
                batch_idx = 0

                batch_count = data_G.batch_count
                for b_g in range(batch_count):
                    x1_y1, x2_y2 = data_G.get_next()
                    x1 = x1_y1[0]
                    y1 = x1_y1[1]
                    x2 = x2_y2[0]
                    y2 = x2_y2[1]
                    true_agreement = np.array(y1 == y2).astype(float)
                    true_agreement = np.reshape(true_agreement, [-1])
                    true_agreement = FT(true_agreement).to(DEVICE)
                    input_x = [x1, x2]
                    pred_agreement = NN(input_x)
                    loss = F.binary_cross_entropy(pred_agreement, true_agreement)
                    loss.backward()

                    optimizer_g.step()
                    record_loss.append(float(loss))
                    batch_idx += 1
                    if batch_idx % log_interval_g == 0:
                        print(
                            'Epoch {}, Batch [g] {} :: Loss {}'.format(
                                epoch, batch_idx, loss)
                        )
                    cur_loss = loss
                    # ------------------------
                    # If training performance is not improving, stop training
                    # ------------------------
                    is_converged, iter_below_tol, = train_utils.check_convergence(
                        prev_loss=prev_loss,
                        cur_loss=loss,
                        cur_step=batch_idx,
                        iter_below_tol=iter_below_tol,
                        abs_loss_chg_tol=0.01,
                        min_num_iter=50,
                        max_iter_below_tol=10
                    )
                    prev_loss = cur_loss
                    if is_converged:
                        final_epoch_g = True
                if final_epoch_g:
                    break

        # -----------------------
        # Train the classifier
        # Use only labelled data
        # ----------------------
        # To do separate out f and g features

        optimizer_f.zero_grad()

        print('[[ --- Training Classifier ---- ]]')
        optimizer_f.zero_grad()

        for epoch in range(num_epochs_f):
            print('Epoch [f]', epoch)

            data_L_generator = singleDataGenerator(
                df_L,
                x_cols=features_F,
                y_col=label_col,
                batch_size=batch_size_r
            )

            data_LL_generator = pairDataGenerator_v2(
                df_1=df_L,
                df_2=df_L,
                x1_F_col=features_F,
                x2_F_col=features_F,
                x1_G_col=features_G,
                x2_G_col=features_G,
                y1_col=None,
                y2_col=label_col,
                batch_size=batch_size_r,
                device=DEVICE,
                allow_refresh=True
            )

            data_UL_generator = pairDataGenerator_v2(
                df_1=df_U,
                df_2=df_L,
                x1_F_col=features_F,
                x2_F_col=features_F,
                x1_G_col=features_G,
                x2_G_col=features_G,
                y1_col=None,
                y2_col=label_col,
                batch_size=batch_size_r,
                device=DEVICE,
                allow_refresh=True
            )

            data_UU_generator = pairDataGenerator_v2(
                df_1=df_U,
                df_2=df_U,
                x1_F_col=features_F,
                x2_F_col=features_F,
                x1_G_col=features_G,
                x2_G_col=features_G,
                y1_col=None,
                y2_col=None,
                batch_size=batch_size_r,
                device=DEVICE,
                allow_refresh=True
            )

            batch_idx_f = 0
            data_L = data_L_generator.get_next()


            while data_L is not None:
                NN.train_mode = 'f'

                # ------  Supervised Loss ------ #
                x1 = data_L[0].to(DEVICE)
                y_true = data_L[1].float().to(DEVICE)
                pred_label = NN(x1)
                loss_s = F.binary_cross_entropy(pred_label, y_true)

                # ====================
                # LL :: lambda_LL * g(x_i,x_j) * d (f(x_i),y_j)
                # ====================

                NN.train_mode = 'f_ll'
                x1_y1, x2_y2 = data_LL_generator.get_next()

                x1_F = x1_y1[0]
                x1_G = x1_y1[1]
                x2_G = x2_y2[1]
                y2 = x2_y2[2]

                pred_agreement, pred_y1 = NN([x1_G, x2_G, x1_F])
                loss_LL = regularization_loss(
                    pred_agreement, [pred_y1, y2]
                )

                # ==================
                # UL
                # ==================
                NN.train_mode = 'f_ul'
                x1_y1, x2_y2 = data_UL_generator.get_next()

                x1_F = x1_y1[0]
                x1_G = x1_y1[1]
                x2_G = x2_y2[1]
                y2 = x2_y2[2]

                pred_agreement, pred_y1 = NN([x1_G, x2_G, x1_F])
                loss_UL = regularization_loss(
                    pred_agreement,
                    [pred_y1, y2]
                )

                # ===================
                # UU
                # ===================
                # print('---- > UU ')
                NN.train_mode = 'f_uu'
                x1_y1, x2_y2 = data_UU_generator.get_next()
                x1_F = x1_y1[0]
                x1_G = x1_y1[1]
                x2_F = x2_y2[0]
                x2_G = x2_y2[1]
                pred_agreement, pred_y1, pred_y2 = NN([x1_G, x2_G, x1_F, x2_F])
                loss_UU = regularization_loss(pred_agreement, [pred_y1, pred_y2])

                # ====================
                # Loss
                # ====================
                loss_total = loss_s + lambda_LL * loss_LL + lambda_UL * loss_UL + lambda_UU * loss_UU
                loss_total.backward()
                optimizer_f.step()
                try:
                    data_L = data_L_generator.get_next()
                except Exception:
                    data_L = None

                batch_idx_f += 1
                if batch_idx_f % log_interval_f == 0:
                    print('Batch[f] {} :: Loss {}'.format(batch_idx_f, loss_total))

        clf_en1_df_U  = None
        clf_en1_features = None
        clf_en1_df_eval = None

        if F_classifier_type == 'MLP':
            cols = [id_col] + features_F + [label_col]
            tmp_df_copy = pd.DataFrame(df[cols],copy=True)
            L_ids = list(df_L_original[id_col])
            U_ids = list(df_U[id_col])

            Y_F_train = (
                tmp_df_copy.loc[
                    tmp_df_copy[id_col].isin(L_ids)
                ][label_col]).values

            del tmp_df_copy[label_col]
            for fg in features_F:
                tmp_df_copy = pd.get_dummies(tmp_df_copy, columns= [fg] )

            X_F = tmp_df_copy.loc[
                    tmp_df_copy[id_col].isin(L_ids)
                ]
            del X_F[id_col]
            X_F = X_F.values

            clf_en1_df_U = tmp_df_copy.loc[
                tmp_df_copy[id_col].isin(U_ids)
            ]
            # Evaluation use
            clf_en1_df_eval = tmp_df_copy.loc[
                tmp_df_copy[id_col].isin(list(df_U_original[id_col]))
            ]

            clf_en1_features = [
                _ for _ in list(clf_en1_df_U) if _ not in list(df_U.columns)
            ]

            clf_en1 = RandomForestClassifier()
            clf_en1.fit(
                X_F, Y_F_train
            )
        else:
            pass

        # ---------------------------
        # Self -labelling
        # ---------------------------

        pred_y_label = []
        pred_y_probs = []



        # Prediction uses multiple classifiers

        self_label_df = clf_en1_df_U.merge(df_U, on = id_col, how = 'inner' )
        self_label_df = self_label_df.sort_values(['score'])
        id_list = list(self_label_df[id_col])


        NN.train(mode=False)
        NN.test_mode = True
        NN.train_mode = False
        idx = 0
        while idx < len(id_list):
            cur_ids =  id_list[idx:idx+512]
            _tmp = self_label_df.loc[self_label_df[id_col].isin(cur_ids)]

            d1 = LT(_tmp[features_F].values).to(DEVICE)
            d2 = _tmp[clf_en1_features].values

            pred_y_probs_1 = NN(d1).cpu().data.numpy()
            pred_y_probs_2 = clf_en1.predict_proba(d2)
            pred_y_probs_1 = np.reshape(pred_y_probs_1, -1)
            pred_y_probs_2 = np.reshape(pred_y_probs_2[:,1], -1)

            _pred_y_probs = np.maximum(
                pred_y_probs_1,
                pred_y_probs_2
            )
            _pred_y_label = np.array(pred_y_probs_2 >= 0.5).astype(int)
            pred_y_label.extend(_pred_y_label)
            pred_y_probs.extend(_pred_y_probs)
            idx += 512
        NN.train(mode=True)
        NN.test_mode = False
        pred_y_probs = np.array(pred_y_probs)


        # ----------------
        # Find the top-k most confident label
        # Update the set of labelled and unlabelled samples
        # ----------------
        df_U_copy = df_U.sort_values(by=['score']).copy()
        k = int(len(df_U) * 0.1)
        self_labelled_samples = train_utils.find_most_confident_samples(
            U_df=df_U_copy,
            y_prob=pred_y_probs,
            threshold=0.4,
            max_count=k
        )
        print(' number of self labelled samples ::', len(self_labelled_samples))

        # remove those ids from df_U
        rmv_id_list = list(self_labelled_samples[id_col])
        df_L = df_L.append(self_labelled_samples, ignore_index=True)
        df_U = df_U.loc[~(df_U[id_col].isin(rmv_id_list))]

        print(' Len of L and U ', len(df_L), len(df_U))
        if len(df_U) < 0.25 * len(df_L):
            continue_training = False

        # Also check for convergence
        current_iter_count += 1

        if current_iter_count > max_IC_iter:
            continue_training = False
        print('----- Validation set ')

        #
        # train_utils.evaluate_validation(
        #     model=NN,
        #     DEVICE=DEVICE,
        #     data_df=df_L_validation,
        #     x_cols=features_F
        # )

        print('----- Test set ')
        pred_y_label = []
        NN.train(mode=False)
        NN.test_mode = True
        NN.train_mode = False

        test_id_list = list(df_U_original[id_col])
        test_df = clf_en1_df_eval.merge(df_U_original, on=id_col, how='inner')
        test_df = test_df.sort_values(['score'])

        true_labels = list(test_df[true_label_col])
        print( true_labels )
        idx = 0
        while idx < len(test_id_list):
            cur_ids = test_id_list[idx:idx + 512]
            idx += 512

            _tmp = test_df.loc[test_df[id_col].isin(cur_ids)]

            d1 = LT(_tmp[features_F].values).to(DEVICE)
            d2 = _tmp[clf_en1_features].values
            pred_y_probs_1 = NN(d1).cpu().data.numpy()
            pred_y_probs_2 = clf_en1.predict_proba(d2)
            pred_y_probs_1 = np.reshape(pred_y_probs_1, -1)
            pred_y_probs_2 = np.reshape(pred_y_probs_2[:,1], -1)
            _pred_y_probs = np.maximum(pred_y_probs_1, pred_y_probs_2)
            _pred_y_label = np.array(_pred_y_probs >= 0.5).astype(int)
            pred_y_label.extend(_pred_y_label)

        NN.train(mode=True)
        NN.test_mode = False
        y_pred = list(pred_y_label)
        y_true = list(np.array(true_labels).astype(int))

        points = [10, 20, 30]
        results_print = pd.DataFrame(
            columns=['next %', 'precision', 'recall', 'f1', 'balanced_accuracy']
        )
        for point in points:
            c = (len(df) * 10 )//100
            _y_pred = y_pred[:c]
            _y_true = y_true[:c]
            b_acc = balanced_accuracy_score(_y_true, _y_true)
            precision = precision_score(_y_true, _y_pred)
            recall = recall_score(_y_true, _y_pred)
            f1 = 2 * precision * recall / (precision + recall)
            print('Next {} % of data ::'.format(point))
            print('Precision ', precision)
            print('Recall ', recall)

            print('f1 ', f1 )
            print('Balanced Accuracy ', b_acc)
            entry_dict = {
                'next %': point,
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1': round(f1, 3),
                'balanced_accuracy': round(b_acc, 3)
            }
            results_print = results_print.append(entry_dict, ignore_index=True)

        #
        # train_utils.evaluate_test(
        #     model=NN,
        #     DEVICE=DEVICE,
        #     data_df=df_U_original,
        #     x_cols=features_F
        # )

    LOGGER.info(results_print.to_string())
    return


# ------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3', 'us_import4', 'us_import5', 'us_import6'],
    default='us_import2'
)

parser.add_argument(
    '--clf', choices=['wide_n_deep', 'deepFM', 'MLP'],
    default='MLP'
)

args = parser.parse_args()
DIR = args.DIR
F_classifier_type =  args.clf

setup_config(DIR)
df_target, normal_data_samples_df, features_F, features_G = data_preprocess.get_data_plus_features(
    DATA_SOURCE_DIR_1,
    DATA_SOURCE_DIR_2,
    model_use_data_DIR,
    F_classifier_type,
    domain_dims,
    serial_mapping_df,
    score_col,
    is_labelled_col,
    label_col,
    true_label_col,
    fraud_col,
    anomaly_col
)

if F_classifier_type == 'wide_n_deep':
    wide_inp_01_dim = len(features_F) - len(features_G)
elif F_classifier_type == 'deepFM':
    wide_inp_01_dim = len(features_F) - len(features_G)

matrix_node_emb = read_matrix_node_emb(matrix_node_emb_path)
node_emb_dim = matrix_node_emb.shape[-1]
num_domains = len(domain_dims)

# matrix_node_emb = FT(matrix_node_emb).to(DEVICE)
matrix_node_emb = FT(matrix_node_emb)
if F_classifier_type == 'MLP':
    dict_clf_initilize_inputs = {
        'mlp_layer_dims': clf_mlp_layer_dimesnions,
        'dropout': 0.05,
        'activation': 'relu'
    }
elif F_classifier_type == 'wide_n_deep':
    dict_clf_initilize_inputs = {}
    dict_clf_initilize_inputs['wide_inp_01_dim'] = wide_inp_01_dim
    dict_clf_initilize_inputs['deep_FC_layer_dims'] = WnD_dnn_layer_dimensions
    dict_clf_initilize_inputs['tune_entity_emb'] = False
elif F_classifier_type == 'deepFM':
    dict_clf_initilize_inputs = {}
    dict_clf_initilize_inputs['wide_inp_01_dim'] = wide_inp_01_dim
    dict_clf_initilize_inputs['dnn_layer_dimensions'] = deepFM_dnn_layer_dimensions
    dict_clf_initilize_inputs['tune_entity_emb'] = False
    dict_clf_initilize_inputs = {
        'mlp_layer_dims': clf_mlp_layer_dimesnions,
        'dropout': 0.05,
        'activation': 'relu'
    }
else:
    dict_clf_initilize_inputs = None

LOGGER.info(' =========== ')
LOGGER.info('F_classifier_type ')

for perc in [10,20,30] :
    NN = SS_network(
        DEVICE,
        node_emb_dimension=node_emb_dim,
        num_domains=num_domains,
        matrix_pretrained_node_embeddings=matrix_node_emb,  # [Number of entities, embedding dimension]
        list_gam_encoder_dimensions=gam_encoder_dimensions_mlp,
        clf_type=F_classifier_type,
        dict_clf_initilize_inputs=dict_clf_initilize_inputs
    )
    if torch.cuda.device_count() > 1:
        print("' >>>> Using ", torch.cuda.device_count(), "GPUs!")
        NN = torch.nn.DataParallel(NN)
        DEVICE = "cuda"
        print(' >>> ', DEVICE)
    NN.to(DEVICE)

    LOGGER.info('Percentage of data labelled {} '.format(perc))
    df_target = train_utils.set_label_in_top_perc(df_target, perc, score_col, true_label_col)
    train_model(NN, df_target, normal_data_samples_df, features_F, features_G)
