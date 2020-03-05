import pandas as pd
import os
import numpy as np
import sys
import pickle
from joblib import Parallel,delayed

# ================================== #

def chunk_df(df, num_chunks):
    chunk_len = int(len(df) // num_chunks)
    list_df_chunks = np.split(
        df.head(chunk_len * (num_chunks - 1)), num_chunks - 1
    )
    end_len = len(df) - chunk_len * (num_chunks - 1)
    list_df_chunks.append(df.tail(end_len))
    return list_df_chunks

# ---------------------------------- #
def aux_generate(
        df,
        neg_samples_df,
        num_neg_samples,
        id_col='PanjivaRecordID'
):
    pos = []
    neg = []
    for i,row in df.iterrows():
        id = row[id_col]
        del row[id_col]
        vals = row.values
        pos.append(vals)
        ns = neg_samples_df.loc[neg_samples_df[id_col]==id].head(num_neg_samples)
        del ns[id_col]
        vals = ns.values
        neg.append(vals)

    pos = np.array(pos)
    neg = np.array(neg)

    return [pos,neg]



def fetch_training_data(
        DATA_DIR,
        training_data_file,
        negative_samples_file,
        num_neg_samples,
        num_jobs = 40,
        id_col = 'PanjivaRecordID'
):
    # Check if files exist !
    pos_data_file = os.path.join(
        DATA_DIR,
        'train_pos_x.npy'
    )
    neg_data_file = os.path.join(
        DATA_DIR,
        'train_neg_x.npy'
    )

    if os.path.exists(pos_data_file) and os.path.exists(neg_data_file):
        pos_x = np.load(pos_data_file)
        neg_x = np.load(neg_data_file)
        return pos_x, neg_x


    train_df = pd.read_csv(
        os.path.join(DATA_DIR, training_data_file)
    )

    neg_samples_df = pd.read_csv(
        os.path.join(DATA_DIR, negative_samples_file)
    )

    list_train_df_chunks = chunk_df (train_df, num_jobs)

    list_res = Parallel(n_jobs=num_jobs)(
        delayed(aux_generate)(
            target_df,
            neg_samples_df,
            num_neg_samples,
        ) for target_df in list_train_df_chunks
    )

    pos = []
    neg = []

    for item in list_res:
        pos.append(item[0])
        neg.append(item[1])

    pos_x = np.vstack(pos)
    neg_x = np.vstack(neg)

    return pos_x, neg_x






