import pickle
import numpy as np
import os
import sys
import inspect
import yaml
import argparse

sys.path.append('./..')
sys.path.append('./../../.')
import pandas as pd

try:
    from src.MEAD import MEAD_model as tf_model
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
DIR = None
DATA_DIR = None
MODEL_OP_FILE_PATH = None
CONFIG_FILE = 'config_1.yaml'
CONFIG = None
DATA_SOURCE_DIR = None

# ----------------------------------------- #

def get_domain_dims():
    global DATA_SOURCE_DIR
    f_path = os.path.join(DATA_SOURCE_DIR, 'domain_dims.pkl')
    with open(f_path, 'rb') as fh:
        res = pickle.load(fh)
    return list(res.values())


# ----------------------------------------- #
# ---------               Model Config
# _RESULT_OP_DIR : Save the output ( dataframe with ids )
# ----------------------------------------- #

def setup_general_config(
        _DIR,
        _RESULT_OP_DIR
):
    global MODEL_NAME
    global DIR
    global MODEL_DIR
    global OP_DIR
    global _SAVE_DIR
    global CONFIG
    global RESULT_OP_DIR

    DIR = _DIR
    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if not os.path.exists(_RESULT_OP_DIR):
        os.mkdir(_RESULT_OP_DIR)
    RESULT_OP_DIR = os.path.join(_RESULT_OP_DIR, DIR)
    if not os.path.exists(RESULT_OP_DIR):
        os.mkdir(RESULT_OP_DIR)

    OP_DIR = os.path.join(CONFIG['OP_DIR'], DIR)
    if not os.path.exists(CONFIG['OP_DIR']):
        os.mkdir(CONFIG['OP_DIR'])
    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    MODEL_DIR = os.path.join(CONFIG['MODEL_DIR'], DIR)
    if not os.path.exists(CONFIG['MODEL_DIR']):
        os.mkdir(os.path.join(CONFIG['MODEL_DIR']))
    if not os.path.exists(MODEL_DIR):
        os.mkdir(os.path.join(MODEL_DIR))
    return


# --------------------------------------------- #

def set_up_model(config, _dir):
    global embedding_dims
    global MODEL_DIR
    global OP_DIR
    global MODEL_NAME
    MODEL_NAME = config['MODEL_NAME']

    if type(config[_dir]['op_dims']) == str:
        embedding_dims = config[_dir]['op_dims']
        embedding_dims = embedding_dims.split(',')
        embedding_dims = [int(e) for e in embedding_dims]
    else:
        embedding_dims = [config[_dir]['op_dims']]

    model_obj = tf_model.model(
        MODEL_NAME,
        MODEL_DIR,
        OP_DIR
    )

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
    global MODEL_DIR

    num_neg_samples = train_x_neg.shape[1]
    CONFIG[DIR]['num_neg_samples'] = num_neg_samples
    model_obj = set_up_model(
        CONFIG,
        DIR
    )

    _use_pretrained = CONFIG[DIR]['use_pretrained']

    if _use_pretrained is True:
        pretrained_file = CONFIG[DIR]['saved_model_file']

        print('Pretrained File :', pretrained_file)
        saved_file_path = os.path.join(
            MODEL_DIR,
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
    id_list = np.reshape(id_list, [-1, 1])

    df_data = np.hstack([id_list, res])
    df = pd.DataFrame(data=df_data, columns=['PanjivaRecordID', 'score'])

    return df


def main():
    global embedding_dims
    global MODEL_DIR
    global DIR
    global DATA_SOURCE_DIR
    global CONFIG
    global CONFIG_FILE
    global MODEL_NAME
    global RESULT_OP_DIR

    DATA_SOURCE_DIR = os.path.join(CONFIG['DATA_SOURCE_DIR'], DIR)
    if not os.path.exists(os.path.join(MODEL_DIR, 'checkpoints')):
        os.mkdir(
            os.path.join(MODEL_DIR, 'checkpoints')
        )

    # ------------ #

    if not os.path.exists(os.path.join(MODEL_DIR, 'checkpoints')):
        os.mkdir(os.path.join(MODEL_DIR, 'checkpoints'))

    train_x_pos, train_x_neg = data_fetcher.get_data_MEAD_train(
        CONFIG['DATA_DIR'],
        DIR
    )

    model_obj = get_trained_model(
        train_x_pos,
        train_x_neg
    )
    scored_df = get_scored_data(model_obj)
    scored_df.to_csv(
        os.path.join( RESULT_OP_DIR,'scored_test_data.csv'),
        index=False
    )

    return scored_df


def  get_scored_data(
    model_obj
):
    global DIR
    global DATA_SOURCE_DIR
    global CONFIG

    test_data_df = data_fetcher.get_Stage2_data_as_DF(
        CONFIG['DATA_DIR'],
        DIR,
        fraud_ratio = CONFIG['fraud_ratio'],
        anomaly_ratio = CONFIG['anomaly_ratio'],
        total_size = CONFIG[DIR]['test_data_size']
    )


    df_copy = test_data_df.copy()
    id_list = list(df_copy['PanjivaRecordID'])
    del df_copy['PanjivaRecordID']
    del df_copy['anomaly']
    del df_copy['fraud']

    data_test_x = df_copy.values

    result_df = score_data(
        model_obj,
        data_test_x,
        id_list
    )
    scores = list(result_df['score'])
    test_data_df['score'] = scores
    test_data_df = test_data_df.sort_values(by=['score'])

    return test_data_df

# ----------------------------------------------------------------- #
# find out which model works best
# ----------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import4', 'us_import5', 'us_import6'],
    default=None
)

parser.add_argument(
    '--RESULT_OP_DIR', default='./../../AD_system_output_v3'
)

args = parser.parse_args()
_DIR = args.DIR
_RESULT_OP_DIR = args.RESULT_OP_DIR
setup_general_config(
    _DIR=_DIR,
    _RESULT_OP_DIR=_RESULT_OP_DIR
)

# main()
test_data_df = data_fetcher.get_Stage2_data_as_DF(
        CONFIG['DATA_DIR'],
        DIR,
        fraud_ratio = CONFIG['fraud_ratio'],
        anomaly_ratio = CONFIG['anomaly_ratio'],
        total_size = CONFIG[DIR]['test_data_size']
    )
print(test_data_df.head(10))
print(list(test_data_df.columns))