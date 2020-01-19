import sys
sys.path.append('./../../.')
sys.path.append('./../.')
import click
import yaml
import os
import pandas as pd

'''
Run 
main.py --DIR ['us_import1', 'us_import2', 'china_import1', 'china_export1']
'''

try:
    from .src.data_preprocessor  import clean_up_test_data
except:
    import clean_up_test_data



def main(DIR):
    CONFIG_FILE = './../config_preprocessor_v02.yaml'

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if DIR is None:
        DIR = CONFIG[DIR]
    data_dir = os.path.join(CONFIG['DATA_DIR'],DIR)
    save_dir = os.path.join(CONFIG['DATA_DIR'],DIR)

    train_df = pd.read_csv(os.path.join(data_dir, CONFIG['train_data_file']),low_memory=False)
    test_df = pd.read_csv(os.path.join(data_dir, CONFIG['test_data_file']),low_memory=False)
    id_col = CONFIG['id_col']

    clean_up_test_data.remove_order1_spurious_coocc(
        train_df,
        test_df,
        id_col
    )
    test_df_file = os.path.join(save_dir, CONFIG['test_data_file'])
    test_df.to_csv(test_df_file, index=False)

    return


@click.command()
@click.option("--DIR", type=click.Choice(['us_import1', 'us_import2', 'china_export', 'china_import'], default = None, case_sensitive=False))





