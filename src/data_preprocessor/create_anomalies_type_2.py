import pandas as pd
import sys
import os
import numpy as np
import pickle
from itertools import combinations
from joblib import Parallel, delayed
from numpy import random
import hashlib
import os
import sys
import pandas as pd
import numpy as np
import sklearn
import glob
import pickle
import random
from joblib import Parallel, delayed
import yaml
import math
from collections import Counter

sys.path.append('.')
sys.path.append('./..')

try:
    from . import utils_createAnomalies as utils_local
except:
    import utils_createAnomalies as utils_local

'''
Find patterns ::
{ A,B } !-> { C } 
'''


def find_conflicting_patterns_aux_1(
        train_df,
        dict_coOccMatrix,
        id_col,
        pattern_size,
        count=100,
        min_normal_pattern_count=5
):
    results = []
    domains = list(sorted(train_df.columns))
    domains.remove(id_col)

    # create set of entity ids for each of the domains
    domain_entitiesSet_dict = {}
    for d in domains:
        domain_entitiesSet_dict[d] = list(set(train_df[d]))

    cur_count = 0
    while cur_count <= count:
        cur_count+=1
        domain_set = np.random.choice(
            domains,
            size=pattern_size,
            replace=False
        )
        excluded_domain = np.random.choice(domain_set, size=1)[0]
        pos_set = list(domain_set)
        pos_set.remove(excluded_domain)
        candidate_dict = {}

        _tries1 = 0

        #  Find { E_d1, E_d2, ... E_d3 } such that they co-occur pairwise
        while True:
            for d in pos_set:
                # sample entity
                candidate_dict[d] = np.random.choice(domain_entitiesSet_dict[d], size=1)[0]

            if utils_local.check_nonZeroCoOccurrence(candidate_dict, dict_coOccMatrix) == True:
                break
            _tries1 += 1

        # ======
        # Find patterns with a minimum "support"
        # ======
        if utils_local.find_pattern_count(candidate_dict, train_df) >= min_normal_pattern_count:
            _tries2 = 0
            max_tries = 1000
            condition_satisfied = False
            cur_domain = excluded_domain

            while not condition_satisfied and _tries2 < max_tries:
                # Select an entity
                cand_e = np.random.choice(domain_entitiesSet_dict[cur_domain], size=1)[0]
                candidate_dict[cur_domain] = cand_e
                # ====
                # Ensure that new candidate has non-zero co-occurrence with others
                for dpair in combinations(list(candidate_dict.keys()), 2):
                    subSet_dict = {}
                    subSet_dict[dpair[0]] = candidate_dict[dpair[0]]
                    subSet_dict[dpair[1]] = candidate_dict[dpair[1]]

                    if not utils_local.check_nonZeroCoOccurrence(subSet_dict, dict_coOccMatrix):
                        condition_satisfied = False
                        break
                    else:
                        condition_satisfied = True
                # ====
                _tries2 += 1
            print("Tries 1 ", _tries1, " Tries 2 ", _tries2)
            results.append(
                candidate_dict
            )
            print('Generated:: ', candidate_dict)

    return results


def generate_anomalies_type_2_aux_2(
        train_df,
        test_df,
        id_col,
        pattern,
        pattern_duplicate_count
):
    # ========
    # select 2 (partial) domains
    # ========
    _domains_set = np.random.choice(
        list(pattern.keys()), replace=False, size=2
    )
    cand = {}
    for d in _domains_set:
        cand[d] = pattern[d]

    # ===
    # Find k records with partial match with pattern
    # ===
    match_df = utils_local.query_df(
        test_df,
        cand
    )
    match_df = match_df.sample(int(1.1 * pattern_duplicate_count))
    new_df = pd.DataFrame(columns=list(train_df.columns))
    for _, row in match_df.iterrows():
        row_copy = pd.Series(row, copy=True)
        for d, e in pattern.items():
            row_copy[d] = e

        new_df = new_df.append(
            row_copy, ignore_index=True
        )
    pattern_idx_str = '002' + str()
    new_df[id_col] = new_df.apply(
        utils_local.aux_modify_id,
        args=(pattern_idx_str,)
    )
    return new_df

# ========== #

def generate_anomalies_type_2(
        train_df,
        test_df,
        save_dir,
        id_col='PanjivaRecordID',
        pattern_size=4,
        reqd_anom_perc=10,
        num_jobs=40,
        pattern_duplicate_count=100
):
    # =====================
    # Over estimating a bit, so that overlaps can be compensated for
    # =====================
    dist_pattern_count = int(1.2 * len(test_df) * (reqd_anom_perc / 100) / 100)
    dict_coOccMatrix = utils_local.get_coOccMatrix_dict(
        train_df,
        id_col
    )
    list_results = Parallel(n_jobs=num_jobs)(
        delayed(find_conflicting_patterns_aux_1)(
            train_df,
            dict_coOccMatrix,
            id_col,
            pattern_size=pattern_size,
            count=dist_pattern_count
        ) for _ in range(num_jobs)
    )
    results = []
    for item in list_results:
        results.extend(item)

    # Remove duplicate 'spurious' patterns generated by the previous call
    patterns = utils_local.dedup_list_dictionaries(
        results
    )

    print(' Percentage of anomalies :: ', len(patterns) / len(test_df))
    result_df = pd.DataFrame(columns=list(train_df.columns))

    # ===================
    # For each pattern create k samples ,
    # where k = pattern_duplicates
    # ===================

    pattern_idx = list(range(len(patterns)))

    res_df_list = Parallel(n_jobs=num_jobs)(
        delayed(generate_anomalies_type_2_aux_2)(
            train_df,
            test_df,
            id_col,
            pattern,
            pattern_duplicate_count
        ) for p_idx, pattern in zip(pattern_idx , patterns)
    )

    # Join the results
    anomalies_df = None
    for _df in res_df_list:
        if anomalies_df is None:
            anomalies_df = pd.DataFrame(_df, copy=True)
        else:
            anomalies_df = anomalies_df.append(_df, ignore_index=True)

    op_path = os.path.join(save_dir, 'anomalies_type3.csv')
    result_df.to_csv(op_path, index=None)
    return result_df

# ========================

