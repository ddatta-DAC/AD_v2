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
from _collections import OrderedDict
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
        reqd_pattern_count,
        support_count,
        company_domain
):
    results = []
    domains = list(sorted(train_df.columns))
    domains.remove(id_col)

    # -------------
    # create set of entity ids for each of the domains
    domain_entitiesSet_dict = {}
    for d in domains:
        domain_entitiesSet_dict[d] = list(set(train_df[d]))
    # -------------

    for cur_count in range(reqd_pattern_count):
        domain_set = [company_domain]
        pick_from = list(domains)
        pick_from.remove(company_domain)
        _domain_set = list(np.random.choice(
            pick_from,
            size=pattern_size-1,
            replace=False
        ))
        # set aside 1 domain(entity)
        excluded_domain = np.random.choice(_domain_set, size=1)[0]
        _domain_set.remove(excluded_domain)
        pos_set = list(_domain_set)
        pos_set.append(company_domain)

        candidate_dict = OrderedDict({})

        _tries1 = 0  # Keep track of counts
        # -------------------
        # Two conditions to be met
        # 1. Find { E_d1, E_d2, ... E_d3 } such that they co-occur pairwise
        # 2. Find patterns with a minimum "support"
        # -------------------
        while True:
            for d in pos_set:   # sample entity
                candidate_dict[d] = np.random.choice(domain_entitiesSet_dict[d], size=1)[0]

            if utils_local.check_nonZeroCoOccurrence(candidate_dict, dict_coOccMatrix)  and \
                utils_local.find_pattern_count(candidate_dict, train_df) >=0:
                break # Found
            _tries1 += 1

        # ------------------------
        # Note :: candidate_dict should now have (ps-1) <domain:entity> where these entities co-occur
        # ------------------------

        max_tries = 10000
        condition_satisfied = False
        cur_domain = excluded_domain
        _tries2 = 0

        while not condition_satisfied and _tries2 < max_tries:
            # Select an entity
            _copy_cd = candidate_dict.copy()
            cand_e = np.random.choice(domain_entitiesSet_dict[cur_domain], size=1)[0]
            _copy_cd[cur_domain] = cand_e
            # ====
            # Ensure that new candidate has non-zero co-occurrence with others
            for dpair in combinations(list(_copy_cd.keys()), 2):
                subSet_dict = {}
                subSet_dict[dpair[0]] = _copy_cd[dpair[0]]
                subSet_dict[dpair[1]] = _copy_cd[dpair[1]]

                if not utils_local.check_nonZeroCoOccurrence(subSet_dict, dict_coOccMatrix):
                    condition_satisfied = False
                    break
            if condition_satisfied == False:
                _tries2 += 1
                continue
            # ensure the first 3rd element does not occur with the first one
            elif utils_local.find_pattern_count(_copy_cd, train_df) == 0:
                candidate_dict = _copy_cd
                condition_satisfied = True
                break
            _tries2 += 1

        # ====
        print("Tries 1 ", _tries1, " Tries 2 ", _tries2)

        if len(candidate_dict) == pattern_size:
            results.append(
                candidate_dict
            )
            # print('Generated:: ', candidate_dict)


    return results


# ====================================
# p_idx : index of the pattern
# pattern : set of items that do not co-occur
# ====================================
def generate_anomalies_type_2_aux_2(
        train_df,
        test_df,
        id_col,
        p_idx,
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
    # upper bound on cluster size
    ns = min(len(match_df), pattern_duplicate_count)
    match_df = match_df.sample(ns)
    new_df = pd.DataFrame(columns=list(train_df.columns))

    for _, row in match_df.iterrows():
        row_copy = pd.Series(row, copy=True)
        for d, e in pattern.items():
            row_copy[d] = e

        new_df = new_df.append(
            row_copy,
            ignore_index=True
        )

    pattern_idx_str = "0020" + str(p_idx)

    print(pattern_idx_str)

    new_df[id_col] = new_df[id_col].apply(
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
        company_domain = None,
        pattern_size=4,
        reqd_anom_perc=50,
        num_jobs=40,
        min_normal_pattern_count=5,
        pattern_duplicate_count=100
):
    # =====================
    # Over estimating a bit, so that overlaps can be compensated for
    # distributed_pattern_count * cluster_size = total anomaly count
    # =====================
    distributed_pattern_count = int(1.05 * len(test_df) * (reqd_anom_perc / 100) / pattern_duplicate_count)

    dict_coOccMatrix = utils_local.get_coOccMatrix_dict(
        train_df,
        id_col
    )

    # =========================
    # List of patterns that are conflicting: {A,B,C}
    # =========================

    list_results = Parallel(n_jobs=num_jobs)(
        delayed(find_conflicting_patterns_aux_1)(
            train_df,
            dict_coOccMatrix,
            id_col,
            pattern_size=pattern_size,
            reqd_pattern_count=distributed_pattern_count,
            support_count=min_normal_pattern_count,
            company_domain = company_domain
        ) for _ in range(num_jobs)
    )
    # Flatten out list to single level
    results = []
    for item in list_results:
        results.extend(item)

    # Remove duplicates in 'spurious' patterns generated by the previous call
    patterns = utils_local.dedup_list_dictionaries(
        results
    )

    print(' Percentage of anomalies :: ', len(patterns) / len(test_df) * 100)
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
            p_idx,
            pattern,
            pattern_duplicate_count
        ) for p_idx, pattern in zip(pattern_idx, patterns)
    )
    return
    # Join the results
    anomalies_df = None
    for _df in res_df_list:
        if anomalies_df is None:
            anomalies_df = pd.DataFrame(_df, copy=True)
        else:
            anomalies_df = anomalies_df.append(_df, ignore_index=True)

    op_path = os.path.join(save_dir, 'anomalies_type2.csv')
    result_df.to_csv(op_path, index=None)
    return result_df

# ========================
