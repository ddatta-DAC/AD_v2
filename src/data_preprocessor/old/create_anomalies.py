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
def get_hash_aux(row, id_col):
    row_dict = row.to_dict()
    del row_dict[id_col]
    _str = '_'.join([str(_) for _ in row_dict.values()])
    _str = str.encode(_str)
    str_hash = hashlib.md5(_str).hexdigest()
    return str_hash


def add_hash(df, id_col):
    df['hash'] = df.apply(
        get_hash_aux,
        axis=1,
        args=(id_col,)
    )
    return df


def is_duplicate(ref_df, hash_val):
    if len(ref_df.loc[ref_df['hash'] == hash_val]) > 0: return True
    return False

# modify the id_col
def aux_modify_id(value, suffix):
    return int(str(value) + str(suffix))


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

            for d, e in generated.items():
                new_row[d] = e

            hash_val = get_hash_aux(new_row, id_col)
            duplicate_flag = is_duplicate(
                ref_df,
                hash_val
            )

            if duplicate_flag == False:
                not_found = False
                break

        anomalies_df = anomalies_df.append(new_row, ignore_index=True)
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

    ref_df = pd.DataFrame(train_df, copy=True)
    ref_df = ref_df.append(test_df, ignore_index=True)
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
        args=('001',)
    )

    op_path = os.path.join(save_dir, 'anomalies_type1.csv')
    anomalies_df.to_csv(op_path, index=None)
    return anomalies_df


'''
Type 2 anomalies

contextual anomaly :
(A,B,C) -> not D  in train set
i.e. A,B,C,D co-occur but D does not co-occur with (A,B,C) 
'''


def find_pattern_count(domainEntity_dict, ref_df):
    global id_col
    query_str = []

    for _c, _i in domainEntity_dict.items():
        query_str.append(' ' + _c + ' == ' + str(_i))

    query_str = ' & '.join(query_str)
    res_query = ref_df.query(query_str)
    return len(res_query)


def aux_func_type_2_3(
        target_df,
        train_df,
        ref_df,
        columnWise_coOccMatrix_dict,
        id_col,
        anom_count,
        pattern_size
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
    max_tries = 5
    max_iterations = 100

    for i, row in working_df.iterrows():
        new_row = pd.Series(row, copy=True)
        # select 3 domains

        iterations = 0
        while iterations < max_iterations:
            domain_set = list(random.choice(domains, replace=False, size=pattern_size))
            trials_1 = 0
            excluded_domain = None
            pos_set = None
            candidate_dict = None

            while trials_1 < max_tries:
                excluded_domain = random.choice(domain_set, size=1)[0]
                pos_set = list(domain_set)
                pos_set.remove(excluded_domain)
                #  Heuristic
                #  Ensure that they co-occur at least 5 times.
                min_pattern_count = 5
                candidate_dict = {}
                for d in pos_set:
                    candidate_dict[d] = row[d]

                if find_pattern_count(candidate_dict, train_df) >= min_pattern_count:
                    break
                trials_1 += 1
            if trials_1 == max_tries:
                continue
            trials_2 = 0
            found = False

            while found == False and trials_2 < max_tries * 3:
                found = False
                candidate_entity = random.choice(domain_entitiesSet_dict[excluded_domain], size=1)[0]
                candidate_dict[excluded_domain] = candidate_entity
                if find_pattern_count(candidate_dict, train_df) == 0:
                    found = True
                # Ensure this is not case of pairwise spurious co-occurrence
                coOcc_nonZero = True
                for entity_pair in combinations(list(candidate_dict.keys()), 2):
                    entity_pair = sorted(entity_pair)
                    key = '_+_'.join(entity_pair)
                    e1 = candidate_dict[entity_pair[0]]
                    e2 = candidate_dict[entity_pair[1]]
                    coOcc_nonZero = (columnWise_coOccMatrix_dict[key][e1][e2] > 0)
                    if coOcc_nonZero == False:
                        found = False
                        break

                if found == True:
                    break
                trials_2 += 1

            # Ensure this does not occur in either train or test set
            hash_val = get_hash_aux(new_row, id_col)
            duplicate_flag = is_duplicate(
                ref_df,
                hash_val
            )

            if duplicate_flag == False:
                break
            iterations += 1
            if iterations == max_iterations:
                print('!! Max iteration reached ... skipping this row !!')

        anomalies_df = anomalies_df.append(new_row, ignore_index=True)
        print(' generated anomaly type 2 or 3')

    return anomalies_df


def generate_type2_anomalies(
        test_df,
        train_df,
        save_dir,
        id_col='PanjivaRecordID',
        num_jobs=40,
        anom_perc=5
):
    print(" :: Generation of anaomalies type 2")
    domains = list(sorted(test_df.columns))
    domains.remove(id_col)

    # Create the  co-occurrence matrix using the reference data frame(training data)
    columnWise_coOccMatrix_dict = get_coOccMatrix_dict(train_df, id_col)

    ref_df = pd.DataFrame(train_df, copy=True)
    ref_df = ref_df.append(test_df, ignore_index=True)
    ref_df = add_hash(ref_df, id_col)

    # chunk data frame :: Parallelize the process
    chunk_len = int(len(test_df) // num_jobs)
    list_df_chunks = np.split(
        test_df.head(chunk_len * (num_jobs - 1)),
        num_jobs - 1
    )

    end_len = len(test_df) - chunk_len * (num_jobs - 1)
    list_df_chunks.append(test_df.tail(end_len))

    distributed_anom_count = int(len(test_df) * anom_perc / 100 * (1 / num_jobs))
    pattern_size = 4
    list_res_df = Parallel(n_jobs=num_jobs)(
        delayed(aux_func_type_2_3)(
            target_df, train_df, ref_df, columnWise_coOccMatrix_dict, id_col, distributed_anom_count, pattern_size
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
        args=('002',)
    )

    op_path = os.path.join(save_dir, 'anomalies_type2.csv')
    anomalies_df.to_csv(op_path, index=None)
    return anomalies_df


def generate_type3_anomalies(
        test_df,
        train_df,
        save_dir,
        id_col='PanjivaRecordID',
        num_jobs=40,
        anom_perc=5
):
    print(" :: Generation of anaomalies type 3")
    domains = list(sorted(test_df.columns))
    domains.remove(id_col)

    # Create the  co-occurrence matrix using the reference data frame(training data)
    columnWise_coOccMatrix_dict = get_coOccMatrix_dict(train_df, id_col)

    ref_df = pd.DataFrame(train_df, copy=True)
    ref_df = ref_df.append(test_df, ignore_index=True)
    ref_df = add_hash(ref_df, id_col)

    # chunk data frame :: Parallelize the process
    chunk_len = int(len(test_df) // num_jobs)
    list_df_chunks = np.split(
        test_df.head(chunk_len * (num_jobs - 1)),
        num_jobs - 1
    )

    end_len = len(test_df) - chunk_len * (num_jobs - 1)
    list_df_chunks.append(test_df.tail(end_len))

    distributed_anom_count = int(len(test_df) * anom_perc / 100 * (1 / num_jobs))
    pattern_size = 3
    list_res_df = Parallel(n_jobs=num_jobs)(
        delayed(aux_func_type_2_3)(
            target_df, train_df, ref_df, columnWise_coOccMatrix_dict, id_col, distributed_anom_count, pattern_size
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
        args=('003',)
    )

    op_path = os.path.join(save_dir, 'anomalies_type3.csv')
    anomalies_df.to_csv(op_path, index=None)
    return anomalies_df
