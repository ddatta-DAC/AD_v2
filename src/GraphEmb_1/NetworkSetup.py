import numpy as np
import pandas as pd
import glob
import os
import sys
import yaml
from pandarallel import pandarallel
import argparse
sys.path.append('./..')
sys.path.append('./../..')
pandarallel.initialize()
# ------------------------------------------ #
try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
    from src.GraphEmb_1 import Random_Walk_v1 as Random_Walk
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher
    from GraphEmb_1 import Random_Walk_v1 as Random_Walk

DIR = None
CONFIG_FILE = 'config_ge_1.yaml'
CONFIG = None
SOURCE_DATA_DIR = None
SAVE_DATA_DIR =  None
id_col = None

# ------------------------------------------ #
# Set up configuration
# ------------------------------------------ #
def set_up_config(_DIR=None):
    global CONFIG
    global SOURCE_DATA_DIR
    global DIR
    global CONFIG_FILE
    global SAVE_DIR
    global id_col
    global SAVE_DATA_DIR

    if _DIR is not None:
        DIR = _DIR
    DATA_DIR = CONFIG
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    SOURCE_DATA_DIR = os.path.join(
        CONFIG['DATA_SOURCE']
    )

    if not os.path.exists(CONFIG['SAVE_DATA_DIR']):
        os.mkdir(CONFIG['SAVE_DATA_DIR'])

    SAVE_DATA_DIR = os.path.join(
        CONFIG['SAVE_DATA_DIR'],
        DIR
    )

    if not os.path.exists(SAVE_DATA_DIR):
        os.mkdir(SAVE_DATA_DIR)

    id_col = CONFIG['id_col']
    return

# ========================================================= #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default=None
)

args = parser.parse_args()
set_up_config(args.DIR)

# ========================================================= #

def get_data():
    global SOURCE_DATA_DIR
    global DIR
    global SAVE_DATA_DIR
    global id_col

    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR, DIR)
    domain_dims = data_fetcher.get_domain_dims(SOURCE_DATA_DIR, DIR)
    print(domain_dims)

    mapping_df_file = 'Serialized_Mapping.csv'
    mapping_df_file = os.path.join(SAVE_DATA_DIR, mapping_df_file)

    if not os.path.exists(mapping_df_file):
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
        print(os.getcwd())
        print(mapping_df_file)
        serial_mapping_df.to_csv(
            mapping_df_file,
            index=False
        )
    else:
        serial_mapping_df = pd.read_csv(mapping_df_file, index_col=None)

    def convert(_row, cols):
        row = _row.copy()
        for c in cols:
            val = row[c]
            _c = c.replace('.1', '')
            res = list(
                serial_mapping_df.loc[
                    (serial_mapping_df['Domain'] == _c) &
                    (serial_mapping_df['Entity_ID'] == val)]
                ['Serial_ID']
            )
            row[c] = res[0]
        return row

    cols = list(data.columns)

    serialized_data = data.parallel_apply(
        convert,
        axis=1,
        args=(cols,)
    )

    return data, serialized_data, domain_dims, serial_mapping_df


MP_list = []
with open('metapaths.txt','r') as fh:
    lines = fh.readlines()
    for line in lines:
        line = line.strip()
        _list = line.split(',')
        MP_list.append(_list)

# --------------------------------------------- #


data, serialized_data, domain_dims, serial_mapping_df = get_data()
rw_obj = Random_Walk.RandomWalkGraph_v1()

rw_obj.initialize(
    data_wdom = data,
    serial_mapping_df =  serial_mapping_df,
    domain_dims = domain_dims,
    id_col = id_col,
    MP_list = MP_list,
    save_data_dir = SAVE_DATA_DIR
)
rw_obj.generate_RandomWalks_w_neg_samples()

