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
except:
    from .src.data_fetcher import data_fetcher_v2 as data_fetcher

DIR = None

CONFIG_FILE = 'config_ge_1.yaml'
CONFIG = None
SOURCE_DATA_DIR = None
SAVE_DIR =  None


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

    SAVE_DIR = os.path.join(
        CONFIG['SAVE_DATA_DIR'],
        DIR
    )

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

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

    records_x , _, _ = data_fetcher.get_data_base_x(SOURCE_DATA_DIR, DIR)
    domain_dims = data_fetcher.get_domain_dims(SOURCE_DATA_DIR, DIR)
    print(domain_dims)
    return records_x, domain_dims

get_data()


