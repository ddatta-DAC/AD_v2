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

try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher

try:
    from . import network_similarity_v1 as NS
except:
    import network_similarity_v1 as NS


# --------------------------------------------

DIR = None
model_use_data_DIR = 'model_use_data'
TARGET_DATA_SOURCE = './../../AD_system_output'

def setup():
    global DIR
    global model_use_data_DIR
    global TARGET_DATA_SOURCE
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

def get_tranasaction_pair_similarity():
    import networkx as nx

    global TARGET_DATA_SOURCE
    domain_dims = get_domain_dims(DIR)
    df = get_training_data(DIR)
    G = NS.get_initial_graph(df, domain_dims)
    test_data_df = read_target_data(
        TARGET_DATA_SOURCE,
        DIR
    )
    G = NS.get_graph_W_transaction_nodes(G,test_data_df)
    print(nx.simrank_similarity(G,10,100))
    return

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
setup()
print (NS.model_use_data_DIR)
domain_dims = get_domain_dims(DIR)
df = get_training_data(DIR)
G = NS.get_initial_graph(df, domain_dims)
df_test = read_target_data()
G = NS.get_graph_W_transaction_nodes(G, df_test)