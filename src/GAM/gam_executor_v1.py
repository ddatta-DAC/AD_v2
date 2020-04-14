#!/usr/bin/env python
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
from pandarallel import pandarallel

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

pandarallel.initialize()
sys.path.append('./../..')
sys.path.append('./..')

import argparse
import multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler, SequentialSampler
import yaml
import pickle
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import tensor
HAS_CUDA = False
DEVICE = None

try:
    if torch.cuda.is_available():
        dev = "cuda:0"
        HAS_CUDA = True
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
    from gam_module import gam_net
    from gam_module import gam_loss
    from clf_net import clf_net_v1 as clf_net
    from clf_net import clf_loss_v1 as clf_loss
    from record_node import graph_net_v1 as graph_net
    from torch_data_loader import pair_Dataset
    from torch_data_loader import type1_Dataset
except:
    from .gam_module import gam_net
    from .gam_module import gam_loss
    from .clf_net import clf_net_v1 as clf_net
    from .clf_net import clf_loss_v1 as clf_loss
    from .record_node import graph_net_v1 as graph_net
    from .torch_data_loader import pair_Dataset
    from .torch_data_loader import type1_Dataset

from torch import FloatTensor as FT
from torch import LongTensor as LT




config_file = 'config.yaml'
CONFIG = None
DATA_SOURCE_DIR_1 = None
DATA_SOURCE_DIR_2 = None
model_use_data_DIR = None
DIR = None
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
epochs_f = 10
epochs_g = 10
log_interval_f = 10
log_interval_g =10
max_IC_iter = 5
clf_mlp_layer_dimesnions = []
# =================================================

def setup_config(_DIR):
    global CONFIG
    global config_file
    global DATA_SOURCE_DIR_1
    global DATA_SOURCE_DIR_2
    global DIR
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

    if _DIR is not None:
        DIR = _DIR

    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)

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
    epochs_g =  CONFIG['epochs_g']
    epochs_f =  CONFIG['epochs_f']
    log_interval_f = CONFIG['log_interval_f']
    log_interval_g = CONFIG['log_interval_g']
    max_IC_iter = CONFIG['max_IC_iter']
    clf_mlp_layer_dimesnions = [
        int(_)
        for _ in CONFIG['classifier_mlp_layers_1'].split(',')
    ]
    print(clf_mlp_layer_dimesnions)
    return

# -------------------------------------


def set_ground_truth_labels(df):
    global true_label_col
    global fraud_col

    def aux_true_label(row):
        if row[fraud_col]:
            return 1
        else:
            return 0

    df[true_label_col] = df.parallel_apply(aux_true_label, axis=1)
    return df


# -----
# Return part of dataframe , where instances are labelled
# -----
def extract_labelled_df(df):
    global is_labelled_col
    res = pd.DataFrame(
        df.loc[df[is_labelled_col] == True],
        copy=True
    )
    return res


def extract_unlabelled_df(df):
    global is_labelled_col
    res = pd.DataFrame(
        df.loc[df[is_labelled_col] == False],
        copy=True
    )
    return res


# -----------------------
# Get o/p from the AD system
# -----------------------
def read_scored_data():
    global score_col
    global DATA_SOURCE_DIR_2
    global label_col

    df = pd.read_csv(
        os.path.join(DATA_SOURCE_DIR_2, 'scored_test_data.csv'), index_col=None
    )
    df = df.sort_values(by=[score_col])
    df[label_col] = 0
    df = set_ground_truth_labels(df)
    return df


def read_matrix_node_emb():
    global node_emb_dim
    global matrix_node_emb_path
    emb = np.load(matrix_node_emb_path)
    node_emb_dim = emb.shape[-1]

    return emb


def set_label_in_top_perc(df, perc):
    global score_col
    global true_label_col
    global id_col
    global is_labelled_col

    df = df.sort_values(by=[score_col])
    if perc > 1:
        perc = perc / 100
    count = int(len(df) * perc)
    df[is_labelled_col] = False

    _tmp = df.head(count)
    cand = list(_tmp[id_col])
    df.loc[df[id_col].isin(cand), label_col] = df.loc[df[id_col].isin(cand), true_label_col]
    df.loc[df[id_col].isin(cand), is_labelled_col] = True

    return df


# --------------------------
# Return the id list of new samples to be aded to labelled set.
# Ensure balance in labelled and unlabelled samples
# --------------------------
def find_most_confident_samples(
        U_df,
        y_probs,  # np.array [?, 2]
        y_pred_label,  # np.array [?,]
        confidence_lb=0.2,
        max_count=None
):
    global label_col
    global id_col
    global is_labelled_col

    pandarallel.initialize()

    if max_count is None:
        max_count = 0.10 * len(U_df)

    # Assuming binary classification : labels are 0 and 1
    y_pred = label_col
    U_df['diff'] = abs(y_probs[:, 0] - y_probs[:, 1])
    U_df[y_pred] = y_pred_label
    valid_flag = 'valid'
    U_df[valid_flag] = False
    U_df_0 = U_df.loc[U_df[y_pred] == 0]
    U_df_1 = U_df.loc[U_df[y_pred] == 1]

    U_df_0 = U_df_0.sort_values(by=['diff'], ascending=False)
    U_df_1 = U_df_1.sort_values(by=['diff'], ascending=False)

    def aux_1(val):
        if val > confidence_lb : return True
        else: return False

    U_df_0[valid_flag] = U_df_0['diff'].apply(aux_1)
    U_df_1[valid_flag] = U_df_1['diff'].apply(aux_1)

    U_df_0 = U_df_0.loc[(U_df_0[valid_flag] == True)]
    U_df_1 = U_df_1.loc[(U_df_1[valid_flag] == True)]
    try:
        del U_df_0['diff']
        del U_df_1['diff']
        del U_df_0[valid_flag]
        del U_df_1[valid_flag]
    except Exception:
        print('ERROR', Exception)
        exit(10)


    count = int(min(min(len(U_df_0), len(U_df_1)), max_count / 2))
    res_df = U_df_1.head(count)
    res_df = res_df.append(U_df_0.head(count), ignore_index=True)
    res_df[is_labelled_col] = True
    return res_df


def convert_to_serial_IDs(
        df,
        keep_entity_ids=False
):
    global feature_col_list
    global serial_mapping_df
    global model_use_data_DIR

    # Save
    f_name = 'data_serializedID_wOrig_' + str(keep_entity_ids) + '.csv'
    f_path = os.path.join(model_use_data_DIR, f_name)
    if os.path.exists(f_path):
        return pd.read_csv(
            f_path,
            index_col=None
        )

    reference_dict = {}
    for d in set(serial_mapping_df['Domain']):
        reference_dict[d] = {}
        _tmp = serial_mapping_df.loc[(serial_mapping_df['Domain'] == d)]
        k = _tmp['Entity_ID']
        v = _tmp['Serial_ID']
        reference_dict[d] = {_k: _v for _k, _v in zip(k, v)}

    # Inplace conversion
    def aux_conv_toSerialID(_row):
        row = _row.copy()
        for fc in feature_col_list:
            col_name = fc
            if keep_entity_ids:
                col_name = '_' + fc
            # row[col_name] = list(
            #     serial_mapping_df.loc[
            #         (serial_mapping_df['Domain'] == fc) &
            #         (serial_mapping_df['Entity_ID'] == row[fc])
            #     ]['Serial_ID'])[0]
            row[col_name] = reference_dict[fc][row[fc]]

        return row

    df = df.parallel_apply(aux_conv_toSerialID, axis=1)
    df.to_csv(f_path, index=False)
    return df


# ============================================== #
# Custom regularization_loss
# ============================================== #
def regularization_loss(g_ij, fi_yj):
    g_ij = g_ij.view(-1)
    val1 = (fi_yj[0] - fi_yj[1]) ** 2
    val2 = val1.float() * g_ij
    val3 = (val2).mean()
    return val3


# -------------------------------------------------- #

'''
1. Train g
2. Train f
3. Add in the most confident labels
'''


class net(nn.Module):
    def __init__(
            self,
            node_emb_dimension,
            num_domains,
            gnet_output_dimensions,
            matrix_pretrained_node_embeddings,
            gam_record_input_dimension,
            gam_encoder_dimensions,
            clf_inp_emb_dimension,
            clf_layer_dimensions
    ):
        super(net, self).__init__()
        # valid values for train_mode are 'f', 'g', False
        self.train_mode = False
        self.test_mode = False
        self.graph_net = None
        self.gam_net = None
        self.clf_net = None

        self.setup_Net(
            node_emb_dimension,
            num_domains,
            gnet_output_dimensions,
            matrix_pretrained_node_embeddings,
            gam_record_input_dimension,
            gam_encoder_dimensions,
            clf_inp_emb_dimension,
            clf_layer_dimensions
        )
        return


    def setup_Net(
            self,
            node_emb_dimension,
            num_domains,
            gnet_output_dimensions,
            matrix_pretrained_node_embeddings,
            gam_record_input_dimension,
            gam_encoder_dimensions,
            clf_inp_emb_dimension,
            clf_layer_dimensions
    ):
        global HAS_CUDA
        global DEVICE

        self.graph_net = graph_net(
            node_emb_dimension,
            num_domains,
            gnet_output_dimensions,
            matrix_pretrained_node_embeddings
        )
        self.gam_net = gam_net(
            gam_record_input_dimension,
            gam_encoder_dimensions,
        )
        self.clf_net = clf_net(
            clf_inp_emb_dimension,
            clf_layer_dimensions
        )

        self.clf_net.to(DEVICE)
        self.gam_net.to(DEVICE)
        self.graph_net.to(DEVICE)

        return


    # ---------------------------
    # Input should be [ Batch, record( list of entities ) ]
    # record( list of entities ) should have serialized entity id
    # ---------------------------
    # input_xy is an list
    def forward(
            self, input_x, input_y=None
    ):
        global HAS_CUDA
        # ----------------
        # Train the agreement module
        # ----------------
        if self.train_mode == 'g':
            x1 = input_x[0]
            x2 = input_x[1]
            # print('[Forward] g ; shapes of x1 and x2 :', x1.shape, x2.shape)
            x1 = self.graph_net(x1)
            x2 = self.graph_net(x2)

            y_pred = self.gam_net(
                x1,
                x2
            )
            return y_pred

        elif self.train_mode == 'f':
            x1 = input_x
            x1 = self.graph_net(x1)
            y_pred = self.clf_net(x1)
            return y_pred
        elif self.train_mode == 'f_ll':
            x1 = input_x[0]
            x2 = input_x[1]
            x1 = self.graph_net(x1)
            x2 = self.graph_net(x2)

            pred_y1 = torch.argmax(
                NN.clf_net(x1),
                dim=1
            )

            pred_agreement = self.gam_net(x1, x2)
            return pred_agreement, pred_y1

        elif self.train_mode == 'f_ul':
            x1 = input_x[0]
            x2 = input_x[1]
            x1 = self.graph_net(x1)
            y1 = self.clf_net(x1)
            x2 = self.graph_net(x2)

            pred_y1 = torch.argmax(y1, dim=1)
            pred_agreement = self.gam_net(x1, x2)
            return pred_agreement, pred_y1

        elif self.train_mode == 'f_uu':
            x1 = input_x[0]
            x2 = input_x[1]
            x1 = self.graph_net(x1)
            x2 = self.graph_net(x2)

            pred_y1 = torch.argmax(self.clf_net(x1), dim=1)
            pred_y2 = torch.argmax(self.clf_net(x2), dim=1)
            pred_agreement = self.gam_net(x1, x2)
            return pred_agreement, pred_y1, pred_y2

        if self.test_mode == True:
            x1 = input_x
            x1 = self.graph_net(x1)
            y_pred = self.clf_net(x1)
            return y_pred

def predict(NN , input_x ):
    NN.train(mode=False)
    NN.test_mode=True
    result = NN(input_x)
    NN.test_mode = False
    NN.train(mode=True)
    return result

# ================================================= #

class dataGeneratorWrapper():

    def __init__(self, obj_dataloader):

        self.obj_dataloader = copy.copy(obj_dataloader)
        self.iter_obj = iter(copy.copy(self.obj_dataloader))
        return

    def generator(self):
        # return next( self.iter_obj )
        for _, batch_data in enumerate():
            yield batch_data

    def get_next(self):
        print('In next...')
        try:
            return next(self.iter_obj)
        except StopIteration:
            print('Encountered StopIteration')
            return None


# ===========================================
# Iterative training
# ===========================================


def train_model(df, NN):
    global epochs_f
    global epochs_g
    global log_interval_f
    global log_interval_g
    global max_IC_iter
    global serialized_feature_col_list
    global feature_col_list
    global DEVICE
    print(' Device :: ',DEVICE )
    batch_size = 256
    num_epochs_g = epochs_g
    num_epochs_f = epochs_f

    num_proc =  multiprocessing.cpu_count()
    lambda_LL = 0.1
    lambda_UL = 0.1
    lambda_UU = 0.05
    current_iter_count = 0
    continue_training = True

    df_L = extract_labelled_df(df)
    df_U = extract_unlabelled_df(df)
    df_U_original = df_U.copy()

    while continue_training:
        # GAM gets inputs as embeddings, which are obtained through the graph embeddings
        # that requires serialized feature ids
        g_feature_cols = serialized_feature_col_list

        NN.train_mode = 'g'

        data_source_L1 = type1_Dataset(
            df_L,
            x_cols=g_feature_cols,
            y_col=label_col
        )

        dataLoader_obj_L1a = DataLoader(
            data_source_L1,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_proc,
            sampler=RandomSampler(data_source_L1),
            drop_last=True
        )
        dataLoader_obj_L1b = DataLoader(
            data_source_L1,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_proc,
            sampler=RandomSampler(data_source_L1),
            drop_last=True
        )
        params_list_g = [_ for _ in NN.graph_net.parameters()]
        params_list_g = params_list_g + ([_ for _ in NN.gam_net.parameters()])
        print('# of parameters to be obtimized for g ', len(params_list_g))
        optimizer_g = torch.optim.Adam(
            params_list_g,
            lr=0.005
        )
        params_list_f = [_ for _ in NN.graph_net.parameters()]
        params_list_f = params_list_f + [_ for _ in NN.gam_net.parameters()]

        optimizer_f = torch.optim.Adam(
            params_list_f,
            lr=0.005
        )
        if NN.train_mode == 'g':
            # ----
            # input_x1,y2 : from Dataloader ( L )
            # input x2,y2 : from Dataloader ( L )
            # For every pair, so nest them
            # -----
            print('Training Agreement model .... ')
            optimizer_g.zero_grad()
            for epoch in range(num_epochs_g):
                print('Epoch [g]', epoch )
                record_loss = []
                batch_idx = 0
                for i, data_i in enumerate(dataLoader_obj_L1a):
                    if type(data_i) == list :
                        data_i = [_.to(DEVICE) for _ in data_i]
                    else:
                        data_i = data_i.to(DEVICE)

                    x1 = data_i[0]
                    y1 = data_i[1]

                    for j, data_j in enumerate(dataLoader_obj_L1b):
                        if type(data_i) == list:
                            data_j = [_.to(DEVICE) for _ in data_j]
                        else:
                            data_j = data_j.to(DEVICE)
                        x2 = data_j[0]
                        y2 = data_j[1]
                        input_x = [x1, x2]

                        true_agreement = np.array(y1 == y2).astype(float)
                        true_agreement = np.reshape(true_agreement, [-1, 1])

                        true_agreement = FT(true_agreement).to(DEVICE)
                        pred_agreement = NN(input_x)

                        loss = gam_loss(pred_agreement, true_agreement)
                        loss.backward()

                        optimizer_g.step()
                        record_loss.append(float(loss))
                        batch_idx += 1
                        if batch_idx % log_interval_g == 0:
                            print(
                                'Epoch {}, Batch [g] {} :: Loss {}'.format(
                                    epoch, batch_idx, loss)
                            )

        # -----------------------
        # Train the classifier
        # Use only labelled data
        # ----------------------
        # To do separate out f and g features

        optimizer_f.zero_grad()

        data_source_L2 = type1_Dataset(
            df_L,
            x_cols=g_feature_cols,
            y_col=label_col
        )

        dataLoader_obj_L2 = DataLoader(
            data_source_L2,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=RandomSampler(data_source_L2),
            drop_last = True
        )
        data_source_LL = pair_Dataset(
            df_L,
            df_L,
            x_cols=g_feature_cols,
            y_col=label_col
        )

        dataLoader_obj_L3 = DataLoader(
            data_source_LL,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=RandomSampler(data_source_LL),
            drop_last=True
        )

        data_source_UL = pair_Dataset(
            df_U,
            df_L,
            x_cols=g_feature_cols,
            y_col=label_col
        )

        dataLoader_obj_L4 = DataLoader(
            data_source_UL,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=RandomSampler(data_source_LL)
        )

        data_source_UU = pair_Dataset(
            df_U,
            df_U,
            x_cols=g_feature_cols,
            y_col=None
        )

        dataLoader_obj_L5 = DataLoader(
            data_source_UU,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=RandomSampler(data_source_UU)
        )

        print('Training Classifier ')
        optimizer_f.zero_grad()
        for epoch in range(num_epochs_f):
            print (' Epoch :: ', epoch)
            # data_L_generator = dataGeneratorWrapper(dataLoader_obj_L2).generator()
            # data_LL_generator = dataGeneratorWrapper(dataLoader_obj_L3).generator()
            # data_UL_generator = dataGeneratorWrapper(dataLoader_obj_L4).generator()
            # data_UU_generator = dataGeneratorWrapper(dataLoader_obj_L5).generator()

            data_L_generator = dataGeneratorWrapper(dataLoader_obj_L2)
            data_LL_generator = dataGeneratorWrapper(dataLoader_obj_L3)
            data_UL_generator = dataGeneratorWrapper(dataLoader_obj_L4)
            data_UU_generator = dataGeneratorWrapper(dataLoader_obj_L5)



            batch_idx_f = 0
            data_L = data_L_generator.get_next()
            while data_L is not None:
                NN.train_mode = 'f'
                # Supervised Loss
                x1 = data_L[0].to(DEVICE)
                y_true = data_L[1].to(DEVICE)
                pred_label = NN(x1)

                loss_s = clf_loss(pred_label, y_true)
                # print(' Loss_s shape', loss_s.shape)
                # ====================
                # LL :: lambda_LL * g(x_i,x_j) * d (f(x_i),y_j)
                # ====================
                # print('---- > LL ')
                NN.train_mode = 'f_ll'
                data_LL_x, data_LL_y = data_LL_generator.get_next()
                x1 = data_LL_x[0].to(DEVICE)
                x2 = data_LL_x[1].to(DEVICE)

                pred_agreement, pred_y1 = NN([x1, x2])
                y2 = LT(data_LL_y[1]).to(DEVICE)

                loss_LL = regularization_loss(
                    pred_agreement, [pred_y1, y2]
                )
                # print(loss_LL.shape)
                # print('---- > UL ')
                # UL
                NN.train_mode = 'f_ul'

                data_UL_x, data_UL_y = data_UL_generator.get_next()

                x1 = data_UL_x[0].to(DEVICE)
                x2 = data_UL_x[1].to(DEVICE)
                y2 = data_UL_y[1].to(DEVICE)
                _x = [x1, x2]

                pred_agreement, pred_y1 = NN(_x)
                loss_UL = regularization_loss(
                    pred_agreement,
                    [pred_y1, y2]
                )
                # print(loss_UL.shape)

                # ====================
                # UU
                # ====================
                # print('---- > UU ')
                NN.train_mode = 'f_uu'
                data_UU = data_UU_generator.get_next()
                x1 = data_UU[0].to(DEVICE)
                x2 = data_UU[1].to(DEVICE)
                _x = [x1, x2]
                pred_agreement, pred_y1, pred_y2 = NN(_x)
                loss_UU = regularization_loss(pred_agreement, [pred_y1, pred_y2])
                # print(loss_UU.shape)

                # ====================
                # Loss
                # ====================
                loss_total = loss_s + lambda_LL * loss_LL + lambda_UL * loss_UL + lambda_UU * loss_UU
                loss_total.backward()
                optimizer_f.step()
                try:
                    data_L = data_L_generator.get_next()
                    # data_L = next(data_L_generator)
                except Exception:
                    print('Exception on iterator', Exception)
                    data_L = None

                batch_idx_f += 1
                if batch_idx_f % log_interval_f == 0:
                    print('Batch[f] {} :: Loss {}'.format(batch_idx_f, loss_total))

        # ---------------------------
        # Self -labelling
        # ---------------------------

        data_source_EU = type1_Dataset(
            df_U,
            x_cols=g_feature_cols,
            y_col=None
        )
        dataLoader_obj_EU = DataLoader(
            data_source_EU,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_proc,
            sampler=SequentialSampler(data_source_EU)
        )

        # data_EU_generator = dataGeneratorWrapper(dataLoader_obj_EU).generator()
        pred_y_label = []
        pred_y_probs = []

        NN.train(mode=False)
        NN.test_mode = True
        NN.train_mode = False
        for batch_idx, data_x in enumerate(dataLoader_obj_EU):
            data_x.to(DEVICE)
            _pred_y_probs = NN(data_x)
            _pred_y_label = torch.argmax(_pred_y_probs, dim=1).cpu().data.numpy()
            _pred_y_probs = _pred_y_probs.cpu().data.numpy()
            pred_y_label.extend(_pred_y_label)
            pred_y_probs.extend(_pred_y_probs)

        NN.train(mode=True)
        NN.test_mode = False
        pred_y_probs = np.array(pred_y_probs)
        pred_y_label = np.array(pred_y_label)

        # ----------------
        # Find the top-k most confident label
        # Update the set of labelled and unlabelled samples
        # ----------------

        k = int(len(df_U) * 0.05)
        self_labelled_samples = find_most_confident_samples(
            U_df=df_U.copy(),
            y_probs=pred_y_probs,
            y_pred_label=pred_y_label,
            confidence_lb=0.20,
            max_count=k
        )
        print( ' number of self labelled samples ::', len(self_labelled_samples))
        # remove those ids from df_U
        rmv_id_list = list(self_labelled_samples[id_col])
        df_L = df_L.append(self_labelled_samples, ignore_index=True)
        df_U = df_U.loc[~(df_U[id_col].isin(rmv_id_list))]

        print(' Len of L and U ', len(df_L), len(df_U))
        if len(df_U) < 0.05 * len(df_L):
            continue_training = False

        # Also check for convergence
        current_iter_count += 1
        if current_iter_count > max_IC_iter:
            continue_training = False

        evaluate_1(
            NN,
            df_U_original,
            x_cols=g_feature_cols
        )
    return


def evaluate_1(
        model,
        data_df,
        x_cols,
        batch_size = 1024
):
    global DEVICE
    global label_col
    global true_label_col
    df = data_df.copy()
    model.train(mode=False)
    model.test_mode = True
    model.train_mode = False

    data_source_eval = type1_Dataset(
        df,
        x_cols=x_cols,
        y_col=None
    )

    dataLoader_obj_eval = DataLoader(
        data_source_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=SequentialSampler(data_source_eval)
    )

    pred_y_label = []
    for batch_idx, data_x in enumerate(dataLoader_obj_eval):
        data_x = data_x.to(DEVICE)
        _pred_y_probs = model(data_x)
        _pred_y_label = torch.argmax(_pred_y_probs, dim=1).cpu().data.numpy()
        pred_y_label.extend(_pred_y_label)

    model.train(mode=True)
    model.test_mode = False
    model.train_mode = True

    pred_y_label = np.array(pred_y_label)
    df[label_col] = list(pred_y_label)

    y_true = df[true_label_col]
    y_pred = df[label_col]
    print('Precision ', precision_score(y_true, y_pred) )
    print('Accuracy ', accuracy_score(y_true, y_pred))
    print('Balanced Accuracy ', balanced_accuracy_score(y_true, y_pred))
    return

# ---------------------------------- #


DIR = 'us_import2'
setup_config(DIR)
df = read_scored_data()
df = convert_to_serial_IDs(df, True)
df = set_label_in_top_perc(df, 10)
matrix_node_emb = read_matrix_node_emb()
num_domains = len(domain_dims)
gam_encoder_dimensions = [512, 512, 256]

# matrix_node_emb = FT(matrix_node_emb).to(DEVICE)
matrix_node_emb = FT(matrix_node_emb)
NN = net(
    node_emb_dimension=node_emb_dim,
    num_domains=num_domains,
    gnet_output_dimensions=node_emb_dim * num_domains,
    matrix_pretrained_node_embeddings=matrix_node_emb,
    gam_record_input_dimension=node_emb_dim * num_domains,
    gam_encoder_dimensions=gam_encoder_dimensions,
    clf_inp_emb_dimension=node_emb_dim * num_domains,
    clf_layer_dimensions=clf_mlp_layer_dimesnions
)


NN.to(DEVICE)
train_model(df, NN)
