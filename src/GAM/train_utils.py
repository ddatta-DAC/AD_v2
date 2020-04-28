# !/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
from sklearn.model_selection import train_test_split
from pandarallel import pandarallel
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


pandarallel.initialize()
import pandas as pd
import os
import pickle

import numpy as np


# -----
# Return part of dataframe , where instances are labelled
# -----
def extract_labelled_df(
    df,
    is_labelled_col='is_labelled'
):
    res = pd.DataFrame(
        df.loc[df[is_labelled_col] == True],
        copy=True
    )
    return res


def extract_unlabelled_df(
    df,
    is_labelled_col='is_labelled'
):
    res = pd.DataFrame(
        df.loc[df[is_labelled_col] == False],
        copy=True
    )
    return res



# ---------------------------------------------------
# Check convergence for classifier for early stopping
# ---------------------------------------------------
def check_convergence(
        prev_loss,
        cur_loss,
        cur_step,
        iter_below_tol,
        abs_loss_chg_tol=0.001,
        min_num_iter=100,
        max_iter_below_tol=50
):
    """Checks if training for a model has converged."""
    has_converged = False

    # Check if we have reached the desired loss tolerance.
    loss_diff = abs(prev_loss - cur_loss)
    if loss_diff < abs_loss_chg_tol:
        iter_below_tol += 1
    else:
        iter_below_tol = 0

    if iter_below_tol >= max_iter_below_tol:
        has_converged = True

    if cur_step < min_num_iter:
        has_converged = False

    return has_converged, iter_below_tol


def obtain_train_validation(
        df,
        split_ratio=0.10
):
    res = train_test_split(
        df,
        test_size=split_ratio
    )

    df_train = res[0]
    df_valid = res[1]

    return df_train, df_valid


# --------------------------
# Return the id list of new samples to be aded to labelled set.
# Ensure balance in labelled and unlabelled samples
# --------------------------
def find_most_confident_samples(
        U_df,  # Data frame
        y_prob,  # np.array [?, 1]
        max_count=None,
        label_col='y',
        is_labelled_col='labelled',
        threshold = 0.4,
        id_col='PanjivaRecordID'
):

    if max_count is None:
        max_count = 0.10 * len(U_df)

    y_pred = label_col

    # Assume output is a probaility value between 0 and 1
    # Calculate p_0 and p_1 : p_0 = 1 - p_1
    col_p_1 = 'y_p_1'
    col_p_0 = 'y_p_0'
    var_col =  '_variance_'
    diff_val = ' _p_diff_'
    valid_flag = '__valid__'

    U_df[diff_val] = np.abs( y_prob - (1 - y_prob) )
    U_df[col_p_1] = y_prob
    U_df[col_p_0] = 1 - y_prob
    U_df[var_col] = y_prob * (1-y_prob)

    # binary classification : labels are 0 and 1
    def _calc_prediction(row):
        if row[col_p_1] > row[col_p_0] : return 1
        else : return 0

    U_df[label_col] = U_df.parallel_apply(_calc_prediction,axis=1)

    U_df[valid_flag] = False
    U_df_0 = pd.DataFrame(U_df.loc[U_df[y_pred] == 0],copy=True)
    U_df_1 = pd.DataFrame(U_df.loc[U_df[y_pred] == 1],copy=True)

    U_df_0 = U_df_0.sort_values(by=[var_col], ascending=False)
    U_df_1 = U_df_1.sort_values(by=[var_col], ascending=False)

    def _set_valid_flag(row):
        return row[diff_val] > threshold

    U_df_0[valid_flag] = U_df_0[diff_val].apply(_set_valid_flag)
    U_df_1[valid_flag] = U_df_1[diff_val].apply(_set_valid_flag)

    # Select Equal number of samples with labels 0 and 1

    U_df_0 = U_df_0.loc[(U_df_0[valid_flag] == True)]
    U_df_1 = U_df_1.loc[(U_df_1[valid_flag] == True)]

    try:
        del U_df_0[var_col]
        del U_df_1[var_col]
        del U_df_0[valid_flag]
        del U_df_1[valid_flag]
        del U_df_1[col_p_0]
        del U_df_1[col_p_1]
    except Exception:
        print('ERROR', Exception)
        exit(1)

    count = int(min(min(len(U_df_0), len(U_df_1)), max_count / 2))
    res_df = U_df_1.head(count)
    res_df = res_df.append(U_df_0.head(count), ignore_index=True)
    res_df[is_labelled_col] = True
    return res_df



def set_label_in_top_perc(
        df,
        perc,
        score_col,
        true_label_col,
        id_col = 'PanjivaRecordID',
        is_labelled_col = 'is_labelled',
        label_col = 'y'
):

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


def evaluate_test(
        model,
        DEVICE,
        data_df,
        x_cols,
        label_col = 'y',
        true_label_col = 'y_true',
        id_col = 'PanjivaRecordID',
        batch_size=3096,

):
    
    df = data_df.copy()
    model.train(mode=False)
    model.test_mode = True
    model.train_mode = False

    data_source_eval = type1_Dataset(
        df,
        x_cols=x_cols,
        y_col=None,
        return_id_col=True
    )

    dataLoader_obj_eval = DataLoader(
        data_source_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=SequentialSampler(data_source_eval)
    )

    id_list = []
    pred_y_label = []
    for batch_idx, data in enumerate(dataLoader_obj_eval):
        _id = data[0].data.numpy()
        _id = np.reshape(_id, -1)
        id_list.extend(_id)
        data_x = data[1].to(DEVICE)
        _pred_y_probs = model(data_x)
        _pred_y_label = torch.argmax(_pred_y_probs, dim=1).cpu().data.numpy()
        pred_y_label.extend(_pred_y_label)

    model.train(mode=True)
    model.test_mode = False
    model.train_mode = True
    pred_y_label = np.array(pred_y_label)

    res_df = pd.DataFrame(
        np.stack([id_list, pred_y_label], axis=1), columns=[id_col, label_col]
    )

    del df[label_col]
    # merge
    df = df.merge(res_df, on=[id_col], how='left')
    # df[label_col] = list(pred_y_label)

    # Now lets ee result at various points
    df = df.sort_values(by=['score'])
    points = [10, 20, 30, 40, 50]
    for point in points:
        print('Next {} % of data ::'.format(point))
        _tmp = df.head(int(len(df) * point / 100))
        y_true = _tmp[true_label_col]
        y_pred = _tmp[label_col]
        print('Precision ', precision_score(y_true, y_pred))
        print('Recall ', recall_score(y_true, y_pred))
        print('Accuracy ', accuracy_score(y_true, y_pred))
        print('Balanced Accuracy ', balanced_accuracy_score(y_true, y_pred))

    return


def evaluate_validation(
        model,
        DEVICE,
        data_df,
        x_cols,
        label_col = 'y',
        true_label_col = 'y_true',
        id_col = 'PanjivaRecordID',
        batch_size=3096
):

    df = data_df.copy()
    model.train(mode=False)
    model.test_mode = True
    model.train_mode = False

    data_source_eval = type1_Dataset(
        df,
        x_cols=x_cols,
        y_col=None,
        return_id_col=True
    )

    dataLoader_obj_eval = DataLoader(
        data_source_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=SequentialSampler(data_source_eval)
    )

    id_list = []
    pred_y_label = []
    for batch_idx, data in enumerate(dataLoader_obj_eval):
        _id = data[0].data.numpy()
        _id = np.reshape(_id, -1)
        id_list.extend(_id)
        data_x = data[1].to(DEVICE)
        _pred_y_probs = model(data_x)
        _pred_y_label = torch.argmax(_pred_y_probs, dim=1).cpu().data.numpy()
        pred_y_label.extend(_pred_y_label)

    model.train(mode=True)
    model.test_mode = False
    model.train_mode = True
    pred_y_label = np.array(pred_y_label)

    res_df = pd.DataFrame(
        np.stack([id_list, pred_y_label], axis=1), columns=[id_col, label_col]
    )

    del df[label_col]
    # merge
    df = df.merge(res_df, on=[id_col], how='left')
    
    # Now lets ee result at various points
    df = df.sort_values(by=['score'])
    y_true = df[true_label_col]
    y_pred = df[label_col]
    print('Precision ', precision_score(y_true, y_pred))
    print('Recall ', recall_score(y_true, y_pred))
    print('Accuracy ', accuracy_score(y_true, y_pred))
    print('Balanced Accuracy ', balanced_accuracy_score(y_true, y_pred))
    return


