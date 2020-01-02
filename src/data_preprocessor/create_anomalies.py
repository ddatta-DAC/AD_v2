import pandas as pd
import sys
import os
import numpy as np
import pickle
from itertools import combinations
from joblib import Parallel, delayed
from numpy import random
import hashlib
try:
    import clean_up_test_data
except:
    from . import clean_up_test_data


# ==========================
# Append a hash to speed up processing
# ==========================
def get_hash_aux(row,id_col):
    row_dict = row.to_dict()
    del row_dict[id_col]
    _str = '_'.join([str(_) for _ in row_dict.values()])
    _str = str.encode(_str)
    str_hash = hashlib.md5(_str).hexdigest()
    return str_hash


def add_hash(df,id_col):

    df['hash'] = None
    df['hash'] = df.apply(
        get_hash_aux,
        axis=1,
        args=(id_col,)
    )
    return df

def check_duplicate(ref_df,  hash_val):
    if len(ref_df.loc[ref_df['hash']==hash_val]) > 0 : return True
    return False


def get_coOccMatrix_dict(df, id_col):
    columns = list(df.columns)
    columns.remove(id_col)
    columns = list(sorted(columns))
    columnWise_coOccMatrix_dict = {}

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_1 = columns[i]
            col_2 = columns[j]
            key = col_1 + '_+_' + col_2
            res = clean_up_test_data.create_coocc_matrix(df, col_1, col_2)
            columnWise_coOccMatrix_dict[key] = res

    return columnWise_coOccMatrix_dict

# modify the id_col
def aux_modify_id( value , suffix ):
    return int(str(value) + str(suffix))


'''
Type 1 :
set of 3 entities which pairwise co-occur, but not present in test or train set
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
        columns= list(target_df.columns)
    )

    for i,row in working_df.iterrows():
        new_row = None
        # generate
        while True:
            domain_set = random.choice(domains, replace=False,  size=3)
            generated = {}
            new_row = pd.Series(row, copy=True)
            for d in domain_set:
                not_satisied = True
                while not_satisied :
                    entity_d = random.choice(
                        domain_entitiesSet_dict[d], size=1
                    )[0]
                    generated[d] = entity_d

                    # Check if selected entities pairwise co-occur
                    if len(generated) > 2:
                        for _pair in combinations(list(generated.keys()),2):
                            _pair = sorted(_pair)
                            key = '_+_'.join(_pair)
                            e1 = generated[_pair[0]]
                            e2 = generated[_pair[1]]
                            not_satisied = (columnWise_coOccMatrix_dict[key][e1][e2] == 0)
                            if not_satisied:
                                break

            for d,e in generated.items():
                new_row[d] = e
            hash_val = get_hash_aux(new_row, id_col)
            is_duplicate = check_duplicate(ref_df, hash_val)
            if is_duplicate == False:
                break

        anomalies_df = anomalies_df.append(new_row,ignore_index=True)
        print(' generated anomaly type 1')
    return anomalies_df


def generate_type1_anomalies(
        test_df,
        train_df,
        save_dir,
        id_col='PanjivaRecordID',
        num_jobs=40,
        anom_perc=10
):
    domains = list(sorted(test_df.columns))
    domains.remove(id_col)

    # Create the  co-occurrence matrix using the reference data frame(training data)
    columnWise_coOccMatrix_dict = get_coOccMatrix_dict(train_df, id_col)

    ref_df = pd.DataFrame(train_df,copy=True)
    ref_df = ref_df.append(test_df,ignore_index=True)
    ref_df = add_hash(ref_df, id_col)

    # chunk data frame :: Parallelize the process
    chunk_len = int(len(test_df) // num_jobs)
    list_df_chunks = np.split(
        test_df.head(chunk_len * (num_jobs - 1)),
        num_jobs - 1
    )

    end_len = len(test_df) - chunk_len * (num_jobs - 1)
    list_df_chunks.append(test_df.tail(end_len))
    print(' Chunk lengths ->', [len(_) for _ in list_df_chunks])
    distributed_anom_count = int(len(test_df) * anom_perc / 100 * (1 / num_jobs))

    list_res_df = Parallel(n_jobs=num_jobs)(
        delayed(aux_func_type_1)(
            target_df, ref_df, columnWise_coOccMatrix_dict, id_col, distributed_anom_count
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
        aux_modify_id,
        args=('001')
    )

    op_path = os.path.join(save_dir, 'anomalies_type1.csv')
    anomalies_df.to_csv(op_path, index=None)
    return anomalies_df
