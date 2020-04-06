import pandas as pd
import numpy as np
import glob
import pickle
import os
import yaml
from pandarallel import pandarallel
import argparse
from multiprocessing import Pool
import multiprocessing
from joblib import Parallel, delayed
import inspect

pandarallel.initialize()
Refresh = True
CONFIG = None
CONFIG_FILE = 'data_loader_config.yaml'
DATA_SOURCE_loc = None
RW_dir = None
Serialized_RW_dir = None
SAVE_DIR_loc = None
flag_REFRESH_create_mp2v_data = False
domain_dims = None

def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    return this_file_path

# ------------------------------------------ #
# Set up configuration
# ------------------------------------------ #
def set_up_config(_DIR = None):
    global CONFIG
    global CONFIG_FILE
    global DATA_SOURCE_loc
    global DIR
    global SAVE_DIR_loc
    global Refresh
    global Serialized_RW_dir
    global RW_dir
    global domain_dims
    global flag_REFRESH_create_mp2v_data
    if _DIR is not None:
        DIR = _DIR

    with open(os.path.join(get_cur_path(), CONFIG_FILE)) as f:
        CONFIG = yaml.safe_load(f)

    if _DIR is None:
        DIR = CONFIG['DIR']
    else:
        DIR = _DIR

    DATA_SOURCE_loc = os.path.join(
        CONFIG['DATA_SOURCE'], DIR
    )
    SAVE_DIR_loc = DATA_SOURCE_loc
    RW_dir = 'RW_Samples'
    Serialized_RW_dir = 'Serialized'

    Refresh = CONFIG[DIR]['Refresh']
    flag_REFRESH_create_data = CONFIG[DIR]['Refresh_create_data']

    with open(
            os.path.join(
                './../../generated_data_v1/',
                DIR,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)

    return

def get_domain_dims():
    global domain_dims
    return domain_dims

# -------
# Data Source : Directory where Random Walks are stored
# This function serializes the ids in a continuous range
# -------
def convert_data(
        DATA_SOURCE=None,
        RW_DIR=None,
        Serialized_DIR=None,
        SAVE_DIR_loc=None,
        domain_dims=None
):
    global Refresh

    mapping_df_file = 'Serialized_Mapping.csv'
    mapping_df_file = os.path.join(
        SAVE_DIR_loc,
        RW_DIR,
        Serialized_DIR,
        mapping_df_file
    )
    RW_SOURCE = os.path.join(
        DATA_SOURCE,
        RW_DIR
    )

    SAVE_DIR = os.path.join(
        SAVE_DIR_loc,
        RW_DIR,
        Serialized_DIR
    )

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not Refresh:
        return SAVE_DIR

    if not os.path.exists(mapping_df_file):
        prev_count = 0
        res = []
        for dn, ds in domain_dims.items():
            for eid in range(ds):
                r = [dn, eid, eid + prev_count]
                res.append(r)
            prev_count += ds

        mapping_df = pd.DataFrame(
            data=res,
            columns= ['Domain', 'Entity_ID', 'Serial_ID']
        )
        print(os.getcwd())
        print(mapping_df_file)
        mapping_df.to_csv(
            mapping_df_file,
            index=False
        )
    else:
        mapping_df = pd.read_csv(mapping_df_file, index_col=None)

    # ----- #
    _files = glob.glob(
        os.path.join(RW_SOURCE, '**.csv')
    )

    mp_specs = sorted([_.split('/')[-1].split('.')[0] for _ in _files])



    def convert(_row, cols):
        row = _row.copy()
        # print(' >> ', row)
        for c in cols:
            val = row[c]
            _c = c.replace('.1', '')
            res = list(
                mapping_df.loc[
                    (mapping_df['Domain'] == _c) &
                    (mapping_df['Entity_ID'] == val)]
                ['Serial_ID']
            )

            row[c] = res[0]

        return row

    for mp_spec in mp_specs:
        old_df = pd.read_csv(
            os.path.join(
                RW_SOURCE,
                mp_spec + '.csv')
        )

        cols = list(old_df.columns)
        print(' old cols ', cols)
        new_df = old_df.parallel_apply(
            convert,
            axis=1,
            args=(cols,)
        )

        # Save file
        df_file_name = mp_spec + '.csv'
        df_file_path = os.path.join(SAVE_DIR, df_file_name)
        new_df.to_csv(
            df_file_path,
            index=None
        )

        # --------------------
        # Read in and convert the negative samples
        # --------------------
        file_name =  mp_spec + '_neg_samples.npy'
        file_path = os.path.join(
            RW_SOURCE,
            file_name
        )
        arr = np.load(file_path)

        file_name = mp_spec + '_serilaized_neg_samples.npy'
        save_file_path = os.path.join(
            SAVE_DIR,
            file_name
        )
        print('Saving to :: ', save_file_path)

        num_samples = arr.shape[1]
        num_cols = len(cols)
        df_ns = pd.DataFrame(
            data = arr.reshape([-1, num_cols]),
            columns = cols
        )

        # convert
        # for i,row in df_ns.iterrows():
        #     print(convert(row,cols))
        pandarallel.initialize()
        new_df_ns = df_ns.parallel_apply(
            convert,
            axis=1,
            args=(cols,)
        )

        data_ns = new_df_ns.values
        data_ns = data_ns.reshape([-1, num_samples, num_cols])
        np.save(save_file_path, data_ns )

    return SAVE_DIR


# -----------------------------------------
# Create ingestion data for metapath2vec model
# ---------------------------------------------------------- #
# Function to create data specific to metapath2vec_1 model
# Following the skip-gram, a word and its context are chosen as well as corresponding negative 'context'
#
# ---------------------------------------------------------- #
def create_data_aux(args) :
    inp_row = args[0]
    row = inp_row.copy()
    _neg_samples_arr = args[1]
    _cols = args[2]
    _ctxt_size = args[3]

    k = _ctxt_size//2
    num_cols = len(_cols)

    centre = []
    context = []
    neg_samples = []

    # from i-k to i+1
    for i in range( k, num_cols-k ):
        _row = row.copy()
        cur_cols = _cols[i-k:i+k+1]
        tmp2 = _neg_samples_arr[:, i-k:i+k+1]

        cur_centre_col = _cols[i]
        cur_centre = _row[cur_centre_col]
        centre.append(cur_centre)
        _row = _row[cur_cols]
        del _row[cur_centre_col]

        context.append(_row.values)

        tmp_df = pd.DataFrame(
            data=tmp2 ,
            columns= cur_cols
        )

        del tmp_df[cur_centre_col]
        ns = tmp_df.values
        neg_samples.append(ns)

    return (centre, context, neg_samples)



def create_metapath2vec_ingestion_data(
    source_file_dir = None,
    model_data_save_dir = None,
    ctxt_size = 2
):
    global flag_REFRESH_create_mp2v_data
    model_data_save_dir = os.path.join(
        source_file_dir,
        model_data_save_dir
    )
    if not os.path.exists(model_data_save_dir):
        os.mkdir(model_data_save_dir)

    if not flag_REFRESH_create_mp2v_data:
        return model_data_save_dir

    print(source_file_dir)
    _files = glob.glob(
        source_file_dir + '/../**.csv'
    )

    mp_specs = sorted([ _.split('/')[-1].split('.')[0] for _ in _files])

    res_centre = []
    res_context = []
    res_neg_samples = []

    for mp_spec in mp_specs :
        df = pd.read_csv(
            os.path.join(source_file_dir, mp_spec + '.csv')
        )

        neg_samples_file = os.path.join(
            source_file_dir, mp_spec+'_serilaized_neg_samples.npy'
        )
        neg_samples = np.load(neg_samples_file)
        neg_samples = neg_samples
        num_jobs = multiprocessing.cpu_count()

        cols = list(df.columns)
        results = Parallel( num_jobs )(
            delayed(create_data_aux)(
                (row, neg_samples[i], cols, ctxt_size),)
            for  i, row in df.iterrows()
        )

        for _result in results :
            _centre = _result[0]
            _context = _result[1]
            _neg_samples = _result[2]

            res_centre.extend(_centre)
            res_context.append(_context)
            res_neg_samples.append(_neg_samples)

    centre = np.array(res_centre)
    context = np.vstack(res_context)
    neg_samples = np.vstack(res_neg_samples)

    print(centre.shape)
    print(context.shape)
    print(neg_samples.shape)

    # -----------------
    # Save data
    # -----------------
    np.save(
        os.path.join(model_data_save_dir, 'x_target.npy'),
        centre
    )
    np.save(
        os.path.join(model_data_save_dir, 'x_context.npy'),
        context
    )
    np.save(
        os.path.join(model_data_save_dir, 'x_neg_samples.npy'),
        neg_samples
    )

    return model_data_save_dir


SAVE_DIR = convert_data(
    DATA_SOURCE=DATA_SOURCE_loc,
    RW_DIR=RW_dir,
    Serialized_DIR=Serialized_RW_dir,
    SAVE_DIR_loc=SAVE_DIR_loc,
    domain_dims=domain_dims
)

mp2v_data_loc = create_metapath2vec_ingestion_data(
    source_file_dir = SAVE_DIR,
    model_data_save_dir = 'metapath2vec_1',
    ctxt_size = 2
)

# ----------------------------------- #


def fetch_model_data_mp2v(
):
    global mp2v_data_loc
    source_dir = mp2v_data_loc
    centre = np.load(os.path.join(source_dir, 'x_target.npy'))
    context = np.load(os.path.join(source_dir, 'x_context.npy'))
    neg_samples = np.load(os.path.join(source_dir, 'x_neg_samples.npy'))
    return centre, context, neg_samples









