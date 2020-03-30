import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')
from joblib import Parallel, delayed
import pickle
import argparse
import multiprocessing
from pandarallel import pandarallel
pandarallel.initialize()
import networkx as nx
import yaml

try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher

try:
    from . import network_similarity as NS
except:
    import network_similarity as NS


# --------------------------------------------

DIR = None
model_use_data_DIR = None
TARGET_DATA_SOURCE = './../../AD_system_output'
CONFIG = None
config_file = 'config.yaml'
KNN_k = None
id_col = 'PanjivaRecordID'
data_max_size = None


def setup():
    global DIR
    global config_file
    global model_use_data_DIR
    global TARGET_DATA_SOURCE
    global KNN_k
    global data_max_size
    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)

    data_max_size = CONFIG['data_max_size']
    KNN_k =  CONFIG['KNN_k']
    model_use_data_DIR = CONFIG['model_use_data_DIR']
    NS.initialize(DIR, model_use_data_DIR)


# ---------------
# Algorithm ::
# Create a network
# Calculate SimRank between the Transaction nodes
#
# With partially labelled data - Train a classifier
# Classify points on the unlabelled data (transaction instances : where features are entities + anomaly scores )
# Set final label as
# Sign ( lambda * Weighted(similarity based) of labels of its K nearest (labelled) neighbors + (1-lambda) predicted label )
# ----------------

def get_training_data(DIR):
    SOURCE_DATA_DIR = './../../generated_data_v1'
    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR, DIR)
    return data


def get_domain_dims(DIR):
    with open(
            os.path.join(
                './../../generated_data_v1/',
                DIR,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


def read_target_data():
    global DIR
    global TARGET_DATA_SOURCE

    csv_f_name = 'scored_test_data.csv'
    df = pd.read_csv(
            os.path.join(
            TARGET_DATA_SOURCE,
            DIR,
            csv_f_name), index_col=None
    )
    return df

# -----------------------------------





# -----------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)
parser.add_argument(
    '--classifier_type', choices=['SVM', 'RF'],
    default='RF'
)
args = parser.parse_args()
DIR = args.DIR
classifier_type = args.classifier_type

# -----------------------------------------
setup()
domain_dims = get_domain_dims(DIR)
df = get_training_data(DIR)
target_df = read_target_data()
target_df = target_df.head(data_max_size)
target_df = target_df.sort_values(
    by = [id_col]
)

record_2_serial_ID = {
    e[0]:e[1] for e in enumerate(list(target_df[id_col]))
}
record_2_serial_ID_df = pd.DataFrame(
    record_2_serial_ID.items(), columns=[id_col,'Serial_ID']
)
record_2_serial_file = os.path.join(model_use_data_DIR, 'record_2_serial_ID.csv')
record_2_serial_ID_df.to_csv(
    record_2_serial_file, index=False
)

NS.initialize(
    DIR,
    model_use_data_DIR
)

NS.process_target_data(
    target_df,
    record_2_serial_ID_df,
    KNN_k
)
