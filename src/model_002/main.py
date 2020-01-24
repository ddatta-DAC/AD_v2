import sys
import os
import yaml
import pandas as pd
import numpy as np
import os
import glob
import pickle
import logging

sys.path.append('./..')
sys.path.append('./../..')
import multiprocessing as mp

try:
    from . import get_embeddings
    from . import context_vector_model_1 as c2v
    from .src.data_fetcher import data_fetcher
    from . import utils_1
except:
    import get_embeddings
    import context_vector_model_1 as c2v
    from src.data_fetcher import data_fetcher
    import utils_1

# ------------------------------- #
CONFIG_FILE = 'config_1.yaml'
DIR = None
OP_DIR = None
modelData_SaveDir = None
DATA_DIR = None
num_jobs = None
CONFIG = None
Refresh_Embeddings = None
logger = None


# ------------------------------- #


def get_logger():
    global OP_DIR
    global DIR
    global CONFIG
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    log_file = os.path.join(OP_DIR, CONFIG['log_file'])
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('=======================================>> ')
    return logger

def close_logger():
    global CONFIG
    global logger
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    return


def setup_config(_DIR=None):
    global CONFIG_FILE
    global DATA_DIR
    global modelData_SaveDir
    global OP_DIR
    global DIR
    global num_jobs
    global Refresh_Embeddings
    global logger
    global CONFIG

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)
    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    DATA_DIR = os.path.join(CONFIG['DATA_DIR'])

    modelData_SaveDir = os.path.join(
        CONFIG['model_data_save_dir'],
        DIR
    )

    if not os.path.exists(CONFIG['OP_DIR']):
        os.mkdir(CONFIG['OP_DIR'])
    OP_DIR = os.path.join(CONFIG['OP_DIR'], DIR)
    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    Refresh_Embeddings = CONFIG[DIR]['Refresh_Embeddings']
    cpu_count = mp.cpu_count()
    num_jobs = min(cpu_count, CONFIG['num_jobs'])

    if not os.path.exists(CONFIG['model_data_save_dir']):
        os.mkdir(CONFIG['model_data_save_dir'])

    if not os.path.exists(modelData_SaveDir):
        os.mkdir(modelData_SaveDir)
    logger = get_logger()
    print(' Set up config')
    return


# ------------------------------------------------ #
def get_entity_embeddings():
    global CONFIG
    global DIR
    global DATA_DIR
    global modelData_SaveDir

    embedding_dims = CONFIG[DIR]['entity_embedding_dims']
    eEmb_num_epochs = CONFIG[DIR]['eEmb_num_epochs']

    # ============== ENTITY EMBEDDING ================ #
    # -------------------------------------------------
    # Check if files exist , if not generate embeddings
    # -------------------------------------------------
    training_data_file = CONFIG['train_data_file']
    files_exist = len(glob.glob(os.path.join(modelData_SaveDir, 'init_embedding**.npy'))) > 0
    if not files_exist or Refresh_Embeddings:
        src_DIR = os.path.join(DATA_DIR, DIR)
        get_embeddings.get_initial_entity_embeddings(
            training_data_file,
            modelData_SaveDir,
            src_DIR,
            embedding_dims,
            eEmb_num_epochs
        )

    # ----- Read in domain_embedding weights ------- #

    domain_emb_wt = []
    file_list = sorted(glob.glob(
        os.path.join(modelData_SaveDir, 'init_embedding**{}.npy'.format(embedding_dims))))
    for npy_file in file_list:
        _tmp_ = np.load(npy_file)
        domain_emb_wt.append(_tmp_)

    return domain_emb_wt


# ================================================ #


def get_domain_dims(dd_file_path):
    with open(dd_file_path, 'rb') as fh:
        domain_dims = pickle.load(fh)
    _tmpDF = pd.DataFrame.from_dict(domain_dims, orient='index')
    _tmpDF = _tmpDF.reset_index()
    _tmpDF = _tmpDF.rename(columns={'index': 'domain'})
    _tmpDF = _tmpDF.sort_values(by=['domain'])
    res = {k: v for k, v in zip(_tmpDF['domain'], _tmpDF[0])}
    return res


# ------------------------------------------------ #

def generate_model_signature():
    global modelData_SaveDir
    signature_prefix = 'model_002_'
    fl = sorted(glob.glob(signature_prefix + '**.h5'))
    if fl is None or len(fl) == 0:
        return signature_prefix + str(1)
    fn = fl[-1]
    idx = fn.split('/')[-1].split('.')[0].split('_')[-1]
    return signature_prefix + str(idx)


# ------------------------------------------------ #
def LOG(_msg):
    global logger
    logger.info(_msg)
    return

# ------------------------------------------------ #
# Get model data
# ------------------------------------------------ #
def get_training_data():
    global DATA_DIR
    global DIR

    train_x_pos, train_x_neg, test_x, test_idList, anomaly_x, _ = data_fetcher.get_data_MEAD(
        DATA_DIR,
        DIR,
        anomaly_type=1
    )
    train_x_pos = np.reshape(train_x_pos, [-1, train_x_pos.shape[1], 1])
    train_x_neg = np.reshape(train_x_neg, [-1, train_x_neg.shape[1], train_x_neg.shape[2], 1])
    return train_x_pos, train_x_neg


# ------------------------------------------------- #
def model_execution():
    global DATA_DIR
    global DIR
    global CONFIG
    global logger

    domain_dims_file = os.path.join(DATA_DIR, DIR, "domain_dims.pkl")
    domain_dims = utils_1.get_domain_dims(domain_dims_file)
    domain_emb_wt = get_entity_embeddings()
    num_domains = len(domain_dims)
    c2v_num_epochs = CONFIG[DIR]['c2v_num_epochs']
    domain_dims_vals = list(domain_dims.values())
    interaction_layer_dim = CONFIG[DIR]['interaction_layer_dim']
    num_neg_samples = CONFIG[DIR]['num_neg_samples']
    lstm_dim = CONFIG[DIR]['lstm_dim']
    context_dim = CONFIG[DIR]['context_dim']
    model_signature = CONFIG[DIR]['saved_model_signature']
    _use_pretrained = model_signature != False
    model_obj = None

    # ------------------------------------------------ #
    if not _use_pretrained:

        train_x_pos, train_x_neg = get_training_data()
        model_signature = generate_model_signature()
        model_obj = c2v.get_model(
            num_domains=num_domains,
            domain_dims=domain_dims_vals,
            domain_emb_wt=domain_emb_wt,
            lstm_dim=lstm_dim,
            interaction_layer_dim=interaction_layer_dim,
            context_dim=context_dim,
            num_neg_samples=num_neg_samples,
            RUN_MODE='train',
            model_signature=model_signature
        )

        # ------------------------------------------------ #
        # Train the model
        # ------------------------------------------------ #
        try:
            model_obj = c2v.model_train(
                model_obj,
                train_x_pos,
                train_x_neg,
                batch_size=512,
                num_epochs=c2v_num_epochs
            )

            model_obj = c2v.save_model(
                modelData_SaveDir,
                model_obj,
                model_signature
            )
            LOG('Trained model :: ' + model_signature)
        except:
            pass


    else:
        print(' Restoring ::', model_signature)
        # Use pretrained model
        model_obj = c2v.get_model(
            num_domains=num_domains,
            domain_dims=domain_dims_vals,
            domain_emb_wt=domain_emb_wt,
            lstm_dim=lstm_dim,
            interaction_layer_dim=interaction_layer_dim,
            context_dim=context_dim,
            num_neg_samples=num_neg_samples,
            RUN_MODE='test',
            save_dir=modelData_SaveDir,
            model_signature=model_signature
        )

setup_config()
model_execution()
close_logger()