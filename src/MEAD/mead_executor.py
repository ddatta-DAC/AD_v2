import operator
import pickle
import numpy as np
import os
import sys
import time
import pprint
import inspect
from collections import OrderedDict
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import auc
import logging
import logging.handlers
sys.path.append('./..')
sys.path.append('./../../.')
import pandas as pd


try:
    from  src.MEAD import MEAD_model as tf_model
except:
    import MEAD_model as tf_model


try:
    from src.Eval import eval_v1 as eval
except:
    from .src.Eval import eval_v1 as eval

try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from .src.data_fetcher import data_fetcher_v2 as data_fetcher

# ------------------------------------ #

cur_path = '/'.join(
    os.path.abspath(
        inspect.stack()[0][1]
    ).split('/')[:-1]
)

sys.path.append(cur_path)

_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "5.0"
__processor__ = 'embedding'
_SAVE_DIR = 'save_dir'
MODEL_NAME = None
_DIR = None
DATA_DIR = None
MODEL_OP_FILE_PATH = None
CONFIG_FILE = 'config_1.yaml'
CONFIG = None


# ----------------------------------------- #

def get_domain_dims():
    global DATA_DIR
    f_path = os.path.join(DATA_DIR, 'domain_dims.pkl')
    with open(f_path, 'rb') as fh:
        res = pickle.load(fh)
    return list(res.values())


# ----------------------------------------- #
# ---------               Model Config    --------- #
# ----------------------------------------- #

def setup_general_config():
    global MODEL_NAME
    global _DIR
    global SAVE_DIR
    global OP_DIR
    global _SAVE_DIR
    global CONFIG
    global logger
    SAVE_DIR = os.path.join(CONFIG['SAVE_DIR'], _DIR)
    OP_DIR = os.path.join(CONFIG['OP_DIR'], _DIR)
    if not os.path.exists(CONFIG['SAVE_DIR']):
        os.mkdir(os.path.join(CONFIG['SAVE_DIR']))

    if not os.path.exists(SAVE_DIR):
        os.mkdir(os.path.join(SAVE_DIR))
    return
# --------------------------------------------- #

def set_up_model(config, _dir):
    global embedding_dims
    global SAVE_DIR
    global OP_DIR
    global MODEL_NAME
    MODEL_NAME = config['MODEL_NAME']

    if type(config[_dir]['op_dims']) == str:
        embedding_dims = config[_dir]['op_dims']
        embedding_dims = embedding_dims.split(',')
        embedding_dims = [int(e) for e in embedding_dims]
    else:
        embedding_dims = [config[_dir]['op_dims']]

    model_obj = tf_model.model(MODEL_NAME, SAVE_DIR, OP_DIR)
    model_obj.set_model_options(
        show_loss_figure=config[_dir]['show_loss_figure'],
        save_loss_figure=config[_dir]['save_loss_figure']
    )

    domain_dims = get_domain_dims()
    LR = config[_dir]['learning_rate']
    model_obj.set_model_hyperparams(
        domain_dims=domain_dims,
        emb_dims=embedding_dims,
        batch_size=config[_dir]['batchsize'],
        num_epochs=config[_dir]['num_epochs'],
        learning_rate=LR,
        num_neg_samples=config[_dir]['num_neg_samples']
    )
    model_obj.set_l2_loss_flag(True)
    model_obj.inference = False
    model_obj.build_model()
    return model_obj

def get_trained_model(
        train_x_pos,
        train_x_neg
):
    global CONFIG
    global logger

    num_neg_samples = train_x_neg.shape[1]
    CONFIG[_DIR]['num_neg_samples'] = num_neg_samples
    model_obj = set_up_model(CONFIG, _DIR)

    _use_pretrained = CONFIG[_DIR]['use_pretrained']

    if _use_pretrained is True:
        pretrained_file = CONFIG[_DIR]['saved_model_file']

        print('Pretrained File :', pretrained_file)
        saved_file_path = os.path.join(
            SAVE_DIR,
            'checkpoints',
            pretrained_file
        )
        if saved_file_path is not None:
            model_obj.set_pretrained_model_file(saved_file_path)
        else:
            model_obj.train_model(
                train_x_pos,
                train_x_neg
            )

    elif _use_pretrained is False:
        model_obj.train_model(
            train_x_pos,
            train_x_neg
        )
    return model_obj

def score_data(
    model_obj,
    data_x,
    id_list
):
    print(' Number of samples ', len(id_list))
    res = model_obj.get_event_score(data_x)
    df_data =  np.vstack([id_list, res])
    df = pd.DataFrame(data = df_data, columns= ['PanjivaRecordID', 'score'])
    df = df.sort_values(by=['score'])
    return df


def main():
    global embedding_dims
    global SAVE_DIR
    global _DIR
    global DATA_DIR
    global CONFIG
    global CONFIG_FILE
    global MODEL_NAME
    global logger

    DATA_DIR = os.path.join(CONFIG['DATA_DIR'], _DIR)
    setup_general_config()


    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(
            os.path.join(SAVE_DIR, 'checkpoints')
        )

    # ------------ #

    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(os.path.join(SAVE_DIR, 'checkpoints'))

    # ------------ #
    logger.info('-------------------')
    logger.info('DIR ' + _DIR)

    train_x_pos, train_x_neg = data_fetcher.get_data_MEAD_train(
        CONFIG['DATA_DIR'],
        _DIR
    )



    model_obj = get_trained_model(
        train_x_pos,
        train_x_neg
    )
    logger.info('------- END ------------')
    # result_df = score_data(
    #     model_obj,
    #     test_data_x,
    #     test_id_list
    # )
    # return result_df







# ----------------------------------------------------------------- #
# find out which model works best
# ----------------------------------------------------------------- #

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

log_file = 'results_v2.log'

_DIR = CONFIG['_DIR']
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
OP_DIR = os.path.join(CONFIG['OP_DIR'], _DIR)

if not os.path.exists(CONFIG['OP_DIR']):
    os.mkdir(CONFIG['OP_DIR'])

if not os.path.exists(OP_DIR):
    os.mkdir(OP_DIR)

handler = logging.FileHandler(os.path.join(OP_DIR, log_file))
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.info(' Info start ')
logger.info(' -----> ' + _DIR)

main()
