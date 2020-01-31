# this is an improvement over APE_V1
# key point : pure unsupervised

# ----------------------- #
# A modified version of APE
# with SGNS instead
# ----------------------- #
import operator
import pickle
import sys
import argparse
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import pandas as pd
import os
import math
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import inspect
import matplotlib.pyplot as plt
import sys
import time
import yaml
import time
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import auc
import glob
import logging
import logging.handlers
tf.set_random_seed(729)

_TIME_IT = False

sys.path.append('./..')
sys.path.append('./../../.')
try:
    from src.Eval import eval_v1 as eval
except:
    from .src.Eval import eval_v1 as eval

try:
    from src.ape import tf_model_ape_1
except:
    import tf_model_ape_1

try:
    from src.data_fetcher import data_fetcher
except:
    from .src.data_fetcher import data_fetcher

# ------------------ #

cur_path = '/'.join(
    os.path.abspath(
        inspect.stack()[0][1]
    ).split('/')[:-2]
)
sys.path.append(cur_path)
FLAGS = tf.app.flags.FLAGS

# ------------------------- #

# ------------------------- #
_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "6.0"


# ------------------------- #


def get_domain_arity():
    global DATA_DIR
    global DIR

    f = os.path.join(DATA_DIR, DIR, 'domain_dims.pkl')

    with open(f, 'rb') as fh:
        dd = pickle.load(fh)
    print(dd)
    return list(dd.values())


def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    print(os.getcwd())
    return this_file_path


# -------- Globals --------- #

# -------- Model Config	  --------- #

logger = None
CONFIG_FILE = 'config_ape_1.yaml'
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)


def setup():

    global SAVE_DIR
    global DIR
    global config
    global OP_DIR
    global MODEL_NAME
    global DATA_DIR
    global domain_dims
    global cur_path
    global logger
    SAVE_DIR = config['SAVE_DIR']
    if DIR is None:
        DIR = config['DIR']
    OP_DIR = config['OP_DIR']
    DATA_DIR = config['DATA_DIR']
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    SAVE_DIR = os.path.join(SAVE_DIR, DIR)

    print(OP_DIR)
    print(SAVE_DIR)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    OP_DIR = os.path.join(OP_DIR, DIR)
    tf_model_ape_1._DIR = DIR

    tf_model_ape_1.OP_DIR = OP_DIR
    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)
    MODEL_NAME = 'model_ape'

    domain_dims = get_domain_arity()
    cur_path = get_cur_path()

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    log_file = os.path.join(OP_DIR, config['log_file'])
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(' Info start ')

# ----------------------------------------- #

# ---------------------------- #
# Return data for training
# x_pos_inp : [?, num_entities]
# x_neg_inp = [?,neg_samples, num_entities]




# --------------------------- #
def main():
    global DIR
    global OP_DIR
    global SAVE_DIR
    global DATA_DIR
    global config
    setup()

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        if os.path.exists(os.path.join(SAVE_DIR, DIR)):
            os.mkdir(os.path.join(SAVE_DIR, DIR))

    checkpoint_dir = os.path.join(SAVE_DIR)
    print(' > ', os.getcwd())

    train_x_pos, train_x_neg, APE_term_2, APE_term_4, _, _,  _,_ = data_fetcher.get_data_APE(
        DATA_DIR,
        DIR
    )
    print(train_x_neg.shape, APE_term_2.shape, APE_term_4.shape)


    domain_dims = data_fetcher.get_domain_dims(DATA_DIR, DIR)
    neg_samples = train_x_neg.shape[1]
    start_time = time.time()
    num_domains = len(domain_dims)
    inp_dims = list(domain_dims.values())

    print('Number of domains ', num_domains )
    print(' domains ', inp_dims)

    model_obj = tf_model_ape_1.model_ape_1(MODEL_NAME)

    model_obj.set_model_params(
        num_entities=num_domains,
        inp_dims=inp_dims,
        neg_samples=neg_samples,
        batch_size=config[DIR]['batch_size'],
        num_epochs=config[DIR]['num_epochs'],
        lr=config[DIR]['learning_rate'],
        chkpt_dir=checkpoint_dir
    )

    _emb_size = int(config[DIR]['embed_size'])
    model_obj.set_hyper_parameters(
        emb_dims=[_emb_size],
        use_bias=[True, False]
    )

    _use_pretrained = config[DIR]['use_pretrained']

    if _use_pretrained is False:
        model_obj.build_model()
        model_obj.train_model(
            train_x_pos,
            train_x_neg,
            APE_term_2,
            APE_term_4
        )

    # test for c = 1, 2, 3

    bounds = []
    training_pos_scores = model_obj.inference(
        train_x_pos
    )
    training_pos_scores = [_[0] for _ in training_pos_scores]

    train_noise = np.reshape(train_x_neg, [-1, train_x_pos.shape[-1]])
    training_noise_scores = model_obj.inference(
        train_noise
    )
    training_noise_scores = [_[0] for _ in training_noise_scores]

    bounds.append(min(training_noise_scores))
    bounds.append(max(training_pos_scores))

    for anomaly_type in [1,2]:
        _, _, _, _, test_x, test_idList, anomaly_x, anomaly_idList = data_fetcher.get_data_APE(
            DATA_DIR,
            DIR,
            anomaly_type
        )

        '''
        join the normal data + anomaly data
        join the normal data id +  anomaly data id 
        Maintain order
        '''

        test_normal_ids = test_idList
        test_anomaly_ids = anomaly_idList
        test_ids = list(np.hstack(
            [test_normal_ids,
             test_anomaly_ids]
        ))
        print (' Len of test_ids ', len(test_ids))
        test_normal_data = test_x
        test_anomaly_data = anomaly_x
        test_data_x = np.vstack([
            test_normal_data,
            test_anomaly_data
        ])

        # ---------- #

        print('Length of test data',test_data_x.shape)
        res = model_obj.inference(
            test_data_x
        )

        test_ids = list(test_ids)
        print('Length of results ', len(res))
        res = list(res)
        _id_score_dict = {
            id: _res for id, _res in zip(test_ids, res)
        }

        '''
        sort by ascending 
        since lower likelihood means anomalous
        '''
        tmp = sorted(
            _id_score_dict.items(),
            key=operator.itemgetter(1)
        )
        sorted_id_score_dict = OrderedDict()

        for e in tmp:
            sorted_id_score_dict[e[0]] = e[1][0]



        recall, precison = eval.precision_recall_curve(
            sorted_id_score_dict,
            anomaly_id_list=test_anomaly_ids
        )

        recall_str = ','.join([str(_) for _ in recall])
        precision_str = ','.join([str(_) for _ in precison])

        logger.info(precision_str)
        logger.info(recall_str)

        _auc = auc(recall, precison)
        logger.info(' Anomaly type' + str(anomaly_type))
        logger.info('AUC')
        logger.info(str(_auc))


        '''
            if _TIME_IT == False:
        
            _auc = auc(recall, precison)
            print('AUC', _auc)
            plt.figure(figsize=[14, 8])
            plt.plot(
                recall,
                precison,
                color='blue', linewidth=1.75)
    
            plt.xlabel('Recall', fontsize=15)
            plt.ylabel('Precision', fontsize=15)
            plt.title('Recall | AUC ' + str(_auc), fontsize=15)
            f_name = 'precison-recall_1_test_' + str(i) + '.png'
            f_path = os.path.join(OP_DIR, f_name)
    
            # plt.savefig(f_path)
            test_result_r.append(recall)
            test_result_p.append(precison)
            plt.close()
        '''

        print('----------------------------')

        end_time = time.time()
        elapsed_time = (end_time - start_time)
        logging.info('time taken')
        logging.info(str(elapsed_time))


parser = argparse.ArgumentParser(description='APE on wwf data')
parser.add_argument('-d','--dir', help='Which data to run. give dir name [ us_import, peru_export, china_export ]', required=True)
args = vars(parser.parse_args())
DIR = None
if 'dir' in args.keys() :
    print(' >>> ', args['dir'])
    DIR = str(args['dir']).strip("'")

main()
