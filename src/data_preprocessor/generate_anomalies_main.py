import pandas as pd
import sys
import os
import yaml
import argparse
import numpy as np
import pickle
from itertools import combinations
from joblib import Parallel, delayed
import multiprocessing as mp
sys.path.append('./')
sys.path.append('./..')

try:
    import create_anomalies_type_1
except:
    from . import create_anomalies_type_1

# ===================

CONFIG_FILE = 'config_preprocessor_v02.yaml'
id_col = 'PanjivaRecordID'
ns_id_col = 'NegSampleID'
num_neg_samples_ape = None
use_cols = None
save_dir = None
company_domain_columns = None
contextual_pattern_support = None
CONFIG = None

# ==================

def set_up_config(_DIR):
    global CONFIG_FILE
    global use_cols
    global DIR
    global save_dir
    global company_domain_columns
    global contextual_pattern_support
    global CONFIG

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)
    if _DIR is None: DIR = CONFIG['DIR']

    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = CONFIG[DIR]['use_cols']
    company_domain_columns = CONFIG[DIR]['company_domain_columns']
    contextual_pattern_support = CONFIG[DIR]['contextual_pattern_support']
    return


# ===================================================== #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'china_export1', 'china_import1'],
    default=None
)

args = parser.parse_args()
DIR = args.DIR
set_up_config(args.DIR)

train_df = pd.read_csv(
    os.path.join(save_dir, CONFIG['train_data_file']),
    index_col=None)
test_df = pd.read_csv(
    os.path.join(save_dir, CONFIG['test_data_file_v1']),
    index_col=None)

# ===================================================== #

num_jobs = CONFIG['num_jobs']
num_jobs = min(mp.cpu_count(),num_jobs)

create_anomalies_type_1.generate_anomalies_type1(
    test_df,
    train_df,
    save_dir,
    id_col=id_col,
    num_jobs=num_jobs,
    anom_perc=100
)

