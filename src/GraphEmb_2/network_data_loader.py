import pandas as pd
import numpy as np
import glob
import pickle
import os
import yaml
from pandarallel import pandarallel
import argparse
from multiprocessing import Pool
import multiprocessing
from joblib import Parallel, delayed
import inspect

pandarallel.initialize()
Refresh = True
CONFIG = None
CONFIG_FILE = 'config.yaml'
DATA_SOURCE_DIR_1 = None
DATA_SOURCE_DIR_2 = None
RW_dir = None
Serialized_RW_dir = None
SAVE_DIR_loc = None
flag_REFRESH_create_mp2v_data = False
domain_dims = None
metapath2vec_data_DIR = None

def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    return this_file_path

# ------------------------------------------ #
# Set up configuration
# ------------------------------------------ #
def set_up_config(_DIR = None):
    global CONFIG
    global CONFIG_FILE
    global DATA_SOURCE_DIR_1
    global DIR
    global Refresh
    global RW_dir
    global domain_dims
    global flag_REFRESH_create_mp2v_data
    global DATA_SOURCE_DIR_2
    global metapath2vec_data_DIR
    if _DIR is not None:
        DIR = _DIR

    with open(os.path.join(get_cur_path(), CONFIG_FILE)) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    DATA_SOURCE_DIR_1 = os.path.join(
        CONFIG['DATA_SOURCE'], DIR
    )

    DATA_SOURCE_DIR_2 = os.path.join(CONFIG['model_use_data_DIR'], DIR)
    RW_dir = CONFIG['RW_Samples_DIR']
    metapath2vec_data_DIR = CONFIG['mp2v_data']
    metapath2vec_data_DIR = os.path.join(DATA_SOURCE_DIR_2, metapath2vec_data_DIR)

    return


# -------
# Data Source : Directory where Random Walks are stored
# This function serializes the ids in a continuous range
# -------
# def convert_data(
#         DATA_SOURCE=None,
#         RW_DIR=None,
#         Serialized_DIR=None,
#         SAVE_DIR_loc=None,
#         domain_dims=None
# ):
#     global Refresh
#
#     mapping_df_file = 'Serialized_Mapping.csv'
#     mapping_df_file = os.path.join(
#         SAVE_DIR_loc,
#         RW_DIR,
#         Serialized_DIR,
#         mapping_df_file
#     )
#     RW_SOURCE = os.path.join(
#         DATA_SOURCE,
#         RW_DIR
#     )
#
#     SAVE_DIR = os.path.join(
#         SAVE_DIR_loc,
#         RW_DIR,
#         Serialized_DIR
#     )
#
#     if not os.path.exists(SAVE_DIR):
#         os.mkdir(SAVE_DIR)
#
#     if not Refresh:
#         return SAVE_DIR
#
#     if not os.path.exists(mapping_df_file):
#         prev_count = 0
#         res = []
#         for dn, ds in domain_dims.items():
#             for eid in range(ds):
#                 r = [dn, eid, eid + prev_count]
#                 res.append(r)
#             prev_count += ds
#
#         mapping_df = pd.DataFrame(
#             data=res,
#             columns= ['Domain', 'Entity_ID', 'Serial_ID']
#         )
#         print(os.getcwd())
#         print(mapping_df_file)
#         mapping_df.to_csv(
#             mapping_df_file,
#             index=False
#         )
#     else:
#         mapping_df = pd.read_csv(mapping_df_file, index_col=None)
#
#     # ----- #
#     _files = glob.glob(
#         os.path.join(RW_SOURCE, '**.csv')
#     )
#
#     mp_specs = sorted([_.split('/')[-1].split('.')[0] for _ in _files])
#
#     def convert(_row, cols):
#         row = _row.copy()
#         # print(' >> ', row)
#         for c in cols:
#             val = row[c]
#             _c = c.replace('.1', '')
#             res = list(
#                 mapping_df.loc[
#                     (mapping_df['Domain'] == _c) &
#                     (mapping_df['Entity_ID'] == val)]
#                 ['Serial_ID']
#             )
#
#             row[c] = res[0]
#
#         return row
#
#     for mp_spec in mp_specs:
#         old_df = pd.read_csv(
#             os.path.join(
#                 RW_SOURCE,
#                 mp_spec + '.csv')
#         )
#
#         cols = list(old_df.columns)
#         print(' old cols ', cols)
#         new_df = old_df.parallel_apply(
#             convert,
#             axis=1,
#             args=(cols,)
#         )
#
#         # Save file
#         df_file_name = mp_spec + '.csv'
#         df_file_path = os.path.join(SAVE_DIR, df_file_name)
#         new_df.to_csv(
#             df_file_path,
#             index=None
#         )
#
#         # --------------------
#         # Read in and convert the negative samples
#         # --------------------
#         file_name =  mp_spec + '_neg_samples.npy'
#         file_path = os.path.join(
#             RW_SOURCE,
#             file_name
#         )
#         arr = np.load(file_path)
#
#         file_name = mp_spec + '_serilaized_neg_samples.npy'
#         save_file_path = os.path.join(
#             SAVE_DIR,
#             file_name
#         )
#         print('Saving to :: ', save_file_path)
#
#         num_samples = arr.shape[1]
#         num_cols = len(cols)
#         df_ns = pd.DataFrame(
#             data = arr.reshape([-1, num_cols]),
#             columns = cols
#         )
#
#         # convert
#         # for i,row in df_ns.iterrows():
#         #     print(convert(row,cols))
#         pandarallel.initialize()
#         new_df_ns = df_ns.parallel_apply(
#             convert,
#             axis=1,
#             args=(cols,)
#         )
#
#         data_ns = new_df_ns.values
#         data_ns = data_ns.reshape([-1, num_samples, num_cols])
#         np.save(save_file_path, data_ns )
#
#     return SAVE_DIR





# SAVE_DIR = convert_data(
#     DATA_SOURCE=DATA_SOURCE_DIR_1,
#     RW_DIR=RW_dir,
#     Serialized_DIR=Serialized_RW_dir,
#     SAVE_DIR_loc=SAVE_DIR_loc,
#     domain_dims=domain_dims
# )



# ----------------------------------- #
# Fetch data specific to metapath2vec model 
# ------------------------------------ #

def fetch_model_data_mp2v():
    global metapath2vec_data_DIR
    source_dir = metapath2vec_data_DIR
    print('Directory :: ', source_dir)
    try:
        centre = np.load(os.path.join(source_dir, 'x_target.npy'))
        context = np.load(os.path.join(source_dir, 'x_context.npy'))
        neg_samples = np.load(os.path.join(source_dir, 'x_neg_samples.npy'))
        return centre, context, neg_samples
    except:
        print('ERROR :: could not find files!')
        exit(1)












