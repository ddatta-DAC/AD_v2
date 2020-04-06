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


DIR = None
config_file = 'config.yaml'
model_use_data_DIR = None
serializedRandomWalk_DIR = None
randomWalk_DIR = None

def set_up_config(_DIR = None):
    global CONFIG
    global config_file
    global DIR
    global model_use_data_DIR
    global serializedRandomWalk_DIR
    global randomWalk_DIR

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

    model_use_data_DIR = CONFIG['model_use_data']
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)

    randomWalk_DIR = CONFIG['randomWalk']
    serializedRandomWalk_DIR = 'Serialized'


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

domain_dims = network_data_loader.get_domain_dims()
num_entities = sum(list(domain_dims.values()))
obj = model_mp2v_1.model()

obj.build(
    emb_dim=128,
    num_entities=num_entities,
    num_neg_samples=10,
    context_size=2,
    batch_size=256,
    num_epochs=10
)

x_t, x_c, x_ns = network_data_loader.fetch_model_data_m2p(
    DIR
)
y = obj.train_model(x_t, x_c, x_ns)
# x = range(len(y))
# plotter.get_general_plot(
#     x,y
# )
