import numpy as np
import pandas as pd
import os
import sys
import yaml
import argparse

sys.path.append('./..')
sys.path.append('./../..')
try:
    from utils import plotter
except:
    from src.utils import plotter

try:
    from .metapath2vec_1 import model_mp2v_1
except:
    from metapath2vec_1 import model_mp2v_1

try:
    from . import network_data_loader
except:
   import network_data_loader

try:
    from . import network_data_loader
except:
   import network_data_loader


try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher


model_name = 'metapath2vec'
DIR = None
config_file = 'config.yaml'
model_use_data_DIR = None
serializedRandomWalk_DIR = None
randomWalk_DIR = None
SOURCE_DATA_DIR_1 = None
SOURCE_DATA_DIR_2 = None

def set_up_config(_DIR = None):
    global CONFIG
    global config_file
    global DIR
    global model_use_data_DIR
    global serializedRandomWalk_DIR
    global randomWalk_DIR
    global model_name
    global model_weights_data
    global SOURCE_DATA_DIR_1
    global SOURCE_DATA_DIR_2
    if _DIR is not None:
        DIR = _DIR

    with open(config_file) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    SOURCE_DATA_DIR_1 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_1'], DIR
    )

    SOURCE_DATA_DIR_2 = os.path.join(
        CONFIG['SOURCE_DATA_DIR_2'], DIR
    )

    model_use_data_DIR = CONFIG['model_use_data_DIR']
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    model_weights_data = CONFIG['model_weights_data']
    if not os.path.exists(model_weights_data):
        os.mkdir(model_weights_data)
    model_weights_data = os.path.join(
        model_weights_data ,DIR , model_name
    )
    metapath2vec_data_DIR = CONFIG['mp2v_data']
    metapath2vec_data_DIR = os.path.join(model_use_data_DIR, metapath2vec_data_DIR)
    return

def get_domain_dims():
    global CONFIG
    global DIR
    return data_fetcher.get_domain_dims(CONFIG['SOURCE_DATA_DIR_1'], DIR)


# --------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default=None
)

args = parser.parse_args()
set_up_config(args.DIR)

# --------------------------------- #

# -------------------------------------------- #

domain_dims = get_domain_dims()
num_entities = sum(list(domain_dims.values()))
model_obj = model_mp2v_1.model()

model_obj.build(
    emb_dim=128,
    num_entities=num_entities,
    num_neg_samples=10,
    context_size=2,
    batch_size=512,
    num_epochs=20
)
# -------------------------------
# Obtain the model training data
# -------------------------------
x_t, x_c, x_ns = network_data_loader.fetch_model_data_mp2v(metapath2vec_data_DIR)
y = model_obj.train_model(
    x_t,
    x_c,
    x_ns
)

# -------------------------------
# Save weights
# -------------------------------
model_obj.save_weights(
    model_weights_data,
    'mp2v_emb.npy'
)





# x = range(len(y))
# plotter.get_general_plot(
#     x,y
# )
