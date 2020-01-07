import pandas as pd
import sys
import os
import numpy as np
import pickle
from itertools import combinations
from joblib import Parallel, delayed
import yaml

sys.path.append('./')
sys.path.append('./..')

try:
    import create_anomalies_type_1
    import create_anomalies_type_2
    import create_anomalies_type_3
except:
    from . import create_anomalies_type_1
    from . import create_anomalies_type_2
    from . import create_anomalies_type_3

# ===================

CONFIG_FILE = 'config_preprocessor_v02.yaml'
id_col = 'PanjivaRecordID'
ns_id_col = 'NegSampleID'
num_neg_samples_ape = None
use_cols = None
save_dir = None
company_domain_columns = None
contextual_pattern_support = None

def set_up_config():
    global CONFIG_FILE
    global use_cols
    global DIR
    global save_dir
    global company_domain_columns
    global contextual_pattern_support

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    DIR = CONFIG['DIR']
    save_dir = os.path.join(
        CONFIG['save_dir'],
        DIR
    )
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    use_cols = CONFIG[DIR]['use_cols']
    company_domain_columns = CONFIG[DIR]['company_domain_columns']
    contextual_pattern_support = CONFIG[DIR]['contextual_pattern_support']
    return CONFIG


CONFIG = set_up_config()
train_df = pd.read_csv(os.path.join(save_dir, CONFIG['train_data_file']))
test_df = pd.read_csv(os.path.join(save_dir, CONFIG['test_data_file_v1']))



# create_anomalies_type_1.generate_anomalies_type1(
#     test_df,
#     train_df,
#     save_dir,
#     id_col='PanjivaRecordID',
#     num_jobs=40,
#     anom_perc=10
# )

# create_anomalies_type_2.generate_anomalies_type_2(
#     train_df,
#     test_df,
#     save_dir,
#     id_col='PanjivaRecordID',
#     pattern_size=3,
#     reqd_anom_perc=10,
#     num_jobs=40,
#     min_normal_pattern_count= contextual_pattern_support,
#     pattern_duplicate_count=100
# )

create_anomalies_type_3.generate_anomalies_type3(
    test_df,
    train_df,
    save_dir,
    id_col='PanjivaRecordID',
    company_domains=company_domain_columns,
    num_jobs=40,
    anom_perc=10,
    cluster_count=100
)
