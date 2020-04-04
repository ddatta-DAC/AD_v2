import numpy as np
import pandas as pd
import glob
import os
import sys
import yaml
import pickle

from pandarallel import pandarallel
import argparse

sys.path.append('./..')
sys.path.append('./../..')
pandarallel.initialize()
# ------------------------------------------ #
try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher

try:
    from . import Random_Walk_v1 as Random_Walk
except:
    import Random_Walk_v1 as Random_Walk
from src.utils import coOccMatrixGenerator as cMg


DIR = None
CONFIG_FILE = 'config.yaml'
CONFIG = None
SOURCE_DATA_DIR = None
SAVE_DATA_DIR = None
id_col = None
model_use_data_DIR = None
serial_mapping_df_file = None
serial_mapping_df = None
SOURCE_DATA_DIR_1 = None
SOURCE_DATA_DIR_2 = None

# ------------------------------------------ #
# Set up configuration
# ------------------------------------------ #
def set_up_config(_DIR=None):
    global CONFIG
    global SOURCE_DATA_DIR_1
    global SOURCE_DATA_DIR_2
    global DIR
    global CONFIG_FILE
    global SAVE_DIR
    global id_col
    global SAVE_DATA_DIR
    global model_use_data_DIR
    global serial_mapping_df_file
    if _DIR is not None:
        DIR = _DIR

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    SOURCE_DATA_DIR_1 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_1']
    )

    SOURCE_DATA_DIR_2 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_2']
    )

    if not os.path.exists(CONFIG['model_use_data_DIR']):
        os.mkdir(CONFIG['model_use_data_DIR'])

    model_use_data_DIR = os.path.join(
        CONFIG['model_use_data_DIR'],
        DIR
    )

    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)

    id_col = CONFIG['id_col']
    mapping_df_file = 'Serialized_Mapping.csv'
    serial_mapping_df_file = os.path.join(model_use_data_DIR, mapping_df_file)

    return






# ========================================================= #
# -----
# Convert to a serial mapping , all entries of a row
# -----
def convert_to_SerialID(_row, cols):
    global serial_mapping_df
    row = _row.copy()
    for c in cols:
        val = row[c]
        res = list(
            serial_mapping_df.loc[
                (serial_mapping_df['Domain'] == c) &
                (serial_mapping_df['Entity_ID'] == val)]
            ['Serial_ID']
        )
        row[c] = res[0]
    return row


def get_coOccMatrixDict(
        df_x
):
    global model_use_data_DIR
    global id_col

    coOccMatrix_File = os.path.join(model_use_data_DIR, 'coOccMatrixSaved.pkl')

    if not os.path.exists(coOccMatrix_File):

        coOCcMatrix_dict = cMg.get_coOccMatrix_dict(df_x, id_col)
        with open(coOccMatrix_File, 'wb') as fh:
            pickle.dump(coOCcMatrix_dict, fh, pickle.HIGHEST_PROTOCOL)
    else:
        with open(coOccMatrix_File, 'rb') as fh:
            coOCcMatrix_dict = pickle.load(fh)
    return coOCcMatrix_dict


def get_data():
    global SOURCE_DATA_DIR_1
    global DIR
    global SAVE_DATA_DIR
    global id_col
    global serial_mapping_df_file
    global serial_mapping_df
    source_1_data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR_1, DIR)
    domain_dims = data_fetcher.get_domain_dims(SOURCE_DATA_DIR_1, DIR)


    if not os.path.exists(serial_mapping_df_file):
        prev_count = 0
        res = []
        for dn, ds in domain_dims.items():
            for eid in range(ds):
                r = [dn, eid, eid + prev_count]
                res.append(r)
            prev_count += ds

        serial_mapping_df = pd.DataFrame(
            data=res,
            columns=['Domain', 'Entity_ID', 'Serial_ID']
        )

        serial_mapping_df.to_csv(
            serial_mapping_df_file,
            index=False
        )
    else:
        serial_mapping_df = pd.read_csv(serial_mapping_df_file, index_col=None)


    return source_1_data


# ========================================================= #

def get_MP_list():
    MP_list = []
    with open('metapaths.txt', 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            line = line.strip()
            _list = line.split(',')
            MP_list.append(_list)
    return MP_list

# --------------------------------------------- #
# Run Random walk generation
# ---------------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)

args = parser.parse_args()
set_up_config(args.DIR)

MP_list = get_MP_list()
domain_dims = data_fetcher.get_domain_dims(SOURCE_DATA_DIR_1, DIR)
source_1_data = get_data()
rw_obj = Random_Walk.RandomWalkGraph_v1()
coOccMatrixDict = get_coOccMatrixDict(source_1_data)

rw_obj.initialize(
    coOccMatrixDict = coOccMatrixDict,
    serial_mapping_df=serial_mapping_df,
    domain_dims=domain_dims,
    id_col=id_col,
    MP_list = MP_list,
    save_data_dir=model_use_data_DIR
)

rw_obj.generate_RandomWalks_w_neg_samples(
    rw_count=10,
    rw_length=120,
    num_neg_samples=10
)
