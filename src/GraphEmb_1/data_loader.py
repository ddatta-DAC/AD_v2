import pandas as pd
import numpy as np
import glob
import pickle
import os
import yaml
from pandarallel import pandarallel
import argparse

pandarallel.initialize()
Refresh = True
CONFIG = None
CONFIG_FILE = 'data_loader_config.yaml'
DATA_SOURCE_loc = None
RW_dir =  None
Serialized_RW_dir = None
SAVE_DIR_loc = None
# ------------------------------------------ #
# Set up configuration
# ------------------------------------------ #
def set_up_config(_DIR=None):
    global CONFIG
    global CONFIG_FILE
    global DATA_SOURCE_loc
    global DIR
    global SAVE_DIR_loc
    global Refresh
    global Serialized_RW_dir
    global RW_dir
    global domain_dims
    
    if _DIR is not None:
        DIR = _DIR

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    DATA_SOURCE_loc = os.path.join(
        CONFIG['DATA_SOURCE'] , DIR
    )
    SAVE_DIR_loc = DATA_SOURCE_loc
    RW_dir = 'RW_Samples'
    Serialized_RW_dir = 'Serialized'
    
    Refresh = CONFIG[DIR]['Refresh']

    with open(
            os.path.join(
                './../../../generated_data_v1/',
                DIR,
                'domain_dims.pkl'
            ),'rb') as fh:
       domain_dims = pickle.load(fh)

    return



# -------
# Data Source : Directory where Random Walks are stored
# This function serializes the ids in a continuous range
# -------
def convert_data(
        DATA_SOURCE = None,
        RW_DIR = None,
        Serialized_DIR = None,
        SAVE_DIR_loc = None,
        domain_dims = None
):
    global Refresh

    mapping_df_file = 'Serialized_Mapping.csv'
    mapping_df_file = os.path.join(SAVE_DIR_loc, mapping_df_file)
    RW_SOURCE = os.path.join(DATA_SOURCE,RW_DIR)

    if not Refresh :
        return
    if not os.path.exists(mapping_df_file):
        prev_count = 0
        res = []
        for dn, ds in domain_dims.items():
            for eid in range(ds):
                r = [dn, eid, eid + prev_count]
                res.append(r)
            prev_count += ds
        
        mapping_df = pd.DataFrame(
            data = res,
            columns =
            ['Domain', 'Entity_ID', 'Serial_ID']
        )
        mapping_df.to_csv(
            mapping_df_file,index=None
        )
    else:
        mapping_df = pd.read_csv(mapping_df_file,index_col = None)
    
    # ----- #
    inp_files = glob.glob(
        os.path.join(RW_SOURCE, '**.csv')
    )

    SAVE_DIR = os.path.join(
        SAVE_DIR_loc,
        RW_DIR,
        Serialized_DIR
    )


    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    for _file in inp_files:
        def convert(row , cols):
            for c in cols:
                _c = c.replace('.1','')
                v = row[c]
                v = list(mapping_df.loc[
                    ( mapping_df['Domain']==_c ) &
                    ( mapping_df['Entity_ID']==v )
                ]['Serial_ID'])
                row[c] = v[0]
            return row
        old_df = pd.read_csv(
            _file, index_col=None
        )
        cols = list(old_df.columns)
        new_df = old_df.parallel_apply(
            convert,
            axis=1,
            args=(cols,)
        )
        
        # Save file
        file_name = _file.split('/')[-1]
        file_path = os.path.join(SAVE_DIR, file_name)
        new_df.to_csv(
            file_path,
            index=None
        )
    return

# --------------------------------------------------------- #
# function to sample
# --------------------------------------------------------- #

# --------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default=None
)

args = parser.parse_args()
set_up_config(args.DIR)




# --------------------------------- #
convert_data(
        DATA_SOURCE = DATA_SOURCE_loc,
        RW_DIR = RW_dir,
        Serialized_DIR = Serialized_RW_dir,
        SAVE_DIR_loc = SAVE_DIR_loc,
        domain_dims = domain_dims
)
