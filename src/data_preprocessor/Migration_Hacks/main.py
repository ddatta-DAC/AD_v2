import sys
import os
import yaml
import pandas as pd
import argparse
sys.path.append('./../../.')
sys.path.append('./../.')

'''
Run 
main.py --DIR ['us_import1', 'us_import2', 'china_import1', 'china_export1']
'''

try:
    from  src.data_preprocessor import clean_up_test_data
except:
    from  data_preprocessor import clean_up_test_data


# ------------------------------------------------ #
data_dir = None
save_dir = None
id_col = None
DIR = None
CONFIG = None
# ------------------------------------------------ #

def set_up_config(_DIR):
    global CONFIG
    global data_dir
    global save_dir
    global  id_col
    global DIR
    CONFIG_FILE = './../config_preprocessor_v02.yaml'
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    DIR = _DIR
    if DIR is None:
        DIR = CONFIG[DIR]
    data_dir = os.path.join('..', CONFIG['save_dir'],DIR)
    save_dir = os.path.join('..', CONFIG['save_dir'],DIR)
    id_col = CONFIG['id_col']
    return


def exec():
    global CONFIG
    global data_dir
    global save_dir
    global id_col
    global DIR
    try:
        train_df = pd.read_csv(os.path.join(data_dir, CONFIG['train_data_file']), index_col=0 ,low_memory=False)
        test_df = pd.read_csv(os.path.join(data_dir, CONFIG['test_data_file']), index_col=0,low_memory=False)
    except:
        train_df = pd.read_csv(os.path.join(data_dir, CONFIG['train_data_file']), low_memory=False)
        test_df = pd.read_csv(os.path.join(data_dir, CONFIG['test_data_file']), low_memory=False)


    clean_up_test_data.remove_order1_spurious_coocc(
        train_df,
        test_df,
        id_col,
        CONFIG['num_jobs']
    )
    test_df_file = os.path.join(save_dir, CONFIG['test_data_file'])
    test_df.to_csv(test_df_file, index=False)
    return

# ------------------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'china_export1', 'china_import1'],
    default=None
)
args = parser.parse_args()
DIR = args.DIR

set_up_config(DIR)
exec()

