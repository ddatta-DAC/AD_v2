import sys
import os
import yaml
import pandas as pd
import numpy as np
import os
import glob
import pickle

try:
    from . import get_embeddings
    from . import context_vector_model_1 as c2v
    from . import fetch_data
except:
    import get_embeddings
    import context_vector_model_1 as c2v
    import fetch_data

CONFIG_FILE = 'config_1.yaml'

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

DIR = CONFIG['DIR']

DATA_DIR = os.path.join(CONFIG['DATA_DIR'], DIR)
training_data_file = os.path.join(
    DATA_DIR,
    CONFIG['train_data_file']
)
model_data_save_dir = os.path.join(
    CONFIG['model_data_save_dir'],
    DIR
)

training_dataDf_file = CONFIG['train_data_file']
negative_samplesDf_file = CONFIG['negative_samples_file']
num_jobs = CONFIG['num_jobs']

# ------------------------------------------------ #

embedding_dims = CONFIG[DIR]['entity_embedding_dims']
domain_dims_file = os.path.join(DATA_DIR, "domain_dims.pkl")
with open(domain_dims_file, 'rb') as fh:
    domain_dims = pickle.load(fh)

num_domains = len(domain_dims)
eEmb_num_epochs = CONFIG[DIR]['eEmb_num_epochs']
c2v_num_epochs = CONFIG[DIR]['c2v_num_epochs']
domain_dims_vals = list(domain_dims.values())
interaction_layer_dim = CONFIG[DIR]['interaction_layer_dim']
num_neg_samples = CONFIG[DIR]['num_neg_samples']
lstm_dim = CONFIG[DIR]['lstm_dim']
context_dim = CONFIG[DIR]['context_dim']
RUN_MODE = None

# ============== ENTITY EMBEDDING ================ #


if not os.path.exists(CONFIG['model_data_save_dir']):
    os.mkdir(CONFIG['model_data_save_dir'])

if not os.path.exists(model_data_save_dir):
    os.mkdir(model_data_save_dir)


# -------------------------------------------------
# Check if files exist , if not generate embeddings
# -------------------------------------------------

files_exist = len(glob.glob(os.path.join(model_data_save_dir, 'init_embedding**.npy'))) > 0
if not files_exist:
    get_embeddings.get_initial_entity_embeddings(
        training_data_file,
        model_data_save_dir,
        DATA_DIR,
        embedding_dims,
        eEmb_num_epochs
    )

# ----- Read in domain_embedding weights ------- #

domain_emb_wt = []
for npy_file in sorted(glob.glob(
        os.path.join(model_data_save_dir, 'init_embedding**.npy')
)):
    _tmp_ = np.load(npy_file)
    domain_emb_wt.append(_tmp_)


# ================================================ #


RUN_MODE = 'train'
model_obj = c2v.get_model(
    num_domains=num_domains,
    domain_dims=domain_dims_vals,
    domain_emb_wt=domain_emb_wt,
    lstm_dim=lstm_dim,
    interaction_layer_dim=interaction_layer_dim,
    context_dim=context_dim,
    num_neg_samples=num_neg_samples,
    RUN_MODE='train'
)

# ------------------------------------------------ #
# Train the model
# ------------------------------------------------ #
pos_x, neg_x = fetch_data.fetch_training_data(
    DATA_DIR,
    training_dataDf_file,
    negative_samplesDf_file,
    num_neg_samples,
    num_jobs
)
model_obj = c2v.model_train(
        model_obj,
        pos_x,
        neg_x,
        batch_size=512,
        num_epochs=c2v_num_epochs
)

model_obj = c2v.save_model(
    model_data_save_dir,
    model_obj
)
# ------------------------------------------------ #
RUN_MODE = 'test'
saved_model_obj = c2v.get_model(
    num_domains=num_domains,
    domain_dims=domain_dims,
    domain_emb_wt=domain_emb_wt,
    lstm_dim=lstm_dim,
    interaction_layer_dim=interaction_layer_dim,
    context_dim=context_dim,
    num_neg_samples=num_neg_samples,
    RUN_MODE='test',
    save_dir=model_data_save_dir
)

c2v.save_model(model_obj)
print(' >>> Model', model_obj.summary())


# ------------------------------------------------ #
