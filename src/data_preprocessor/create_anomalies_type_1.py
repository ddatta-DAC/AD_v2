import pandas as pd
import sys
import os
import numpy as np
import pickle
from itertools import combinations
from joblib import Parallel, delayed
from numpy import random
import hashlib
sys.path.append('./')

try:
    from . import utils_createAnomalies as utils_local
    from . import clean_up_test_data
except:
    import utils_createAnomalies as utils_local
    import clean_up_test_data


'''
Type 1 :
set of 3 entities which pairwise do not co-occur, and not present in test or train set
'''


def aux_func_type_1(
        target_df,
        ref_df,
        columnWise_coOccMatrix_dict,
        id_col,
        anom_count
):
    working_df = target_df.sample(anom_count)
    domains = list(sorted(target_df.columns))
    domains.remove(id_col)
    # create set of entity ids for each of the domains
    domain_entitiesSet_dict = {}

    for d in domains:
        domain_entitiesSet_dict[d] = list(set(ref_df[d]))

    anomalies_df = pd.DataFrame(
        columns=list(target_df.columns)
    )

    for i, row in working_df.iterrows():
        new_row = None
        # generate
        not_found = True
        tries_1 = 0

        while not_found:
            domain_set = random.choice(domains, replace=False, size=3)
            generated = {}
            new_row = pd.Series(row, copy=True)
            not_satisied = True

            while not_satisied:

                for d in domain_set:
                    entity_d = random.choice(
                        domain_entitiesSet_dict[d], size=1
                    )[0]
                    generated[d] = entity_d

                # Check if selected entities pairwise co-occur
                for _pair in combinations(list(generated.keys()), 2):
                    _pair = sorted(_pair)
                    key = '_+_'.join(_pair)
                    e1 = generated[_pair[0]]
                    e2 = generated[_pair[1]]
                    is_zero = (columnWise_coOccMatrix_dict[key][e1][e2] == 0)
                    if is_zero == False:
                        not_satisied = True
                        break
                    else:
                        not_satisied = False

                tries_1 +=1
            print('Tries :: ',tries_1)
            for d, e in generated.items():
                new_row[d] = e

            hash_val = utils_local.get_hash_aux(new_row, id_col)
            duplicate_flag = utils_local.is_duplicate(
                ref_df,
                hash_val
            )

            if duplicate_flag == False:
                print('Generated :: ', new_row[id_col] )
                break

        anomalies_df = anomalies_df.append(
            new_row,
            ignore_index=True
        )
        print(' generated anomaly type 1')

    return anomalies_df


def generate_anomalies_type1(
        test_df,
        train_df,
        save_dir,
        id_col='PanjivaRecordID',
        num_jobs=40,
        anom_perc=10
):
    domains = list(sorted(test_df.columns))
    domains.remove(id_col)

    # Create the co-occurrence matrix using the reference data frame(training data)
    columnWise_coOccMatrix_dict = utils_local.get_coOccMatrix_dict(train_df, id_col)
    ref_df = pd.DataFrame(train_df, copy=True)
    ref_df = ref_df.append(test_df, ignore_index=True)
    ref_df = utils_local.add_hash(ref_df, id_col)

    # chunk data frame :: Parallelize the process
    list_df_chunks = utils_local.chunk_df(test_df, num_jobs)
    print(' Chunk lengths ->', [len(_) for _ in list_df_chunks])
    distributed_anom_count = int(len(test_df) * anom_perc / 100 * (1 / num_jobs))
    print('Anomalies generation per job :: ',distributed_anom_count )
    list_res_df = Parallel(n_jobs=num_jobs)(
        delayed(aux_func_type_1)(
            target_df,
            ref_df,
            columnWise_coOccMatrix_dict,
            id_col,
            distributed_anom_count
        ) for target_df in list_df_chunks
    )
    print('Post cleaning chunk lengths ->', [len(_) for _ in list_res_df])

    anomalies_df = None
    for _df in list_res_df:
        if anomalies_df is None:
            anomalies_df = _df
        else:
            anomalies_df = anomalies_df.append(_df, ignore_index=True)

    anomalies_df = anomalies_df.drop_duplicates(subset=domains)
    anomalies_df[id_col] = anomalies_df[id_col].apply(
        utils_local.aux_modify_id,
        args=('001',)
    )

    op_path = os.path.join(save_dir, 'anomalies_type1.csv')
    anomalies_df.to_csv(op_path, index=None)
    return anomalies_df
