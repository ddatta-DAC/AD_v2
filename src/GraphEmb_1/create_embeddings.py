import numpy as np
import pandas as pd
import glob
import os
import sys
import yaml
from pandarallel import pandarallel
pandarallel.initialize()
import argparse
# ------------------------------------------ #
try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
    from src.GraphEmb_1 import Random_Walk_v1 as Random_Walk
except:
    from .src.data_fetcher import data_fetcher_v2 as data_fetcher
    from .src.GraphEmb_1 import Random_Walk_v1 as Random_Walk

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
    records_x = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR, DIR)
    domain_dims = data_fetcher.get_domain_dims(SOURCE_DATA_DIR, DIR)
    print(domain_dims)
    return records_x, domain_dims

records_x, domain_dims = get_data()
rw_obj = Random_Walk.RandomWalkGraph_v1()
MP_list = [
    ['ShipperPanjivaID', 'PortOfLading', 'Carrier', 'PortOfUnlading', 'ConsigneePanjvaID'],
    ['ShipperPanjivaID', 'ShipmentOrigin', 'HSCode', 'ShipmentDestination', 'ConsigneePanjvaID'],
    ['ShipperPanjivaID', 'PortOfLading', 'HSCode', 'PortOfUnlading', 'ConsigneePanjvaID'],
    ['ShipmentOrigin', 'PortOfLading', 'PortOfUnlading', 'ShipmentDestination'],
    ]

rw_obj.initialize(
    records_x,
    domain_dims,
    id_col,
    MP_list = MP_list,
    save_data_dir = SAVE_DATA_DIR
)
rw_obj.generate_RandomWalks()




