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
Keep size to 3
'''


def find_conflicting_patterns_aux_1(
        train_df,
        dict_coOccMatrix,
        id_col,
        reqd_pattern_count,
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
        pick_from = list(domains)

        # select the next domain
        domain_1 = company_domain
        pick_from.remove(domain_1)
        domain_2 = np.random.choice(pick_from, size=1, replace=False)[0]
        pick_from.remove(domain_2)

        e1 = np.random.choice(domain_entitiesSet_dict[domain_1], size=1)[0]
        candidate_dict = None
        max_trials_1 = 200
        max_trials_2 = 1000
        trials_1 = 0
        found = False
        while trials_1 <= max_trials_1:
            trials_1 += 1
            # select an entity from domain2
            e2 = np.random.choice(domain_entitiesSet_dict[domain_2], size=1)[0]
            tmp_dict = {
                domain_1: e1,
                domain_2: e2
            }
            if not utils_local.check_nonZeroCoOccurrence(tmp_dict, dict_coOccMatrix):
                continue
            # find the 3rd domain and entity
            # such that (a,b , C) do not co-occur
            # but a,c, b,c co-occur
            trials_2 = 0

            domain_3 = np.random.choice(pick_from, size=1, replace=False)[0]
            # Find entities that do not co-occur with e1 and e2
            tmpDf = train_df.loc[(train_df[domain_1] == e1) & (train_df[domain_2] == e2)]
            rmvE_list = list(tmpDf[domain_3])
            _enity_set = set(domain_entitiesSet_dict[domain_3]).difference(set(rmvE_list))
            _enity_set = list(_enity_set)
            while trials_2 <= max_trials_2:
                trials_2 += 1
                e3 = np.random.choice(_enity_set, size=1)[0]

                # check
                _tmp1 = {
                    domain_1: e1,
                    domain_3: e3
                }
                cond1 = utils_local.check_nonZeroCoOccurrence(tmp_dict, dict_coOccMatrix)
                if not cond1:
                    continue

                _tmp2 = {
                    domain_2: e2,
                    domain_3: e3
                }
                cond2 = utils_local.check_nonZeroCoOccurrence(tmp_dict, dict_coOccMatrix)
                if not cond2:
                    continue

                _tmp3 = OrderedDict({
                    domain_1: e1,
                    domain_2: e2,
                    domain_3: e3
                })
                if utils_local.find_pattern_count(_tmp3, train_df) == 0:
                    candidate_dict = _tmp3
                    found = True
                    break
            if found: break

        if candidate_dict is not None:
            results.append(candidate_dict)

    return results


# ====================================
# p_idx : index of the pattern
# pattern : set of items that do not co-occur ; it is OrderedDict
# ====================================
def generate_anomalies_type_2_aux_2(
        train_df,
        test_df,
        id_col,
        p_idx,
        pattern,
        pattern_cluster_size
):
    # ========
    # select first 2 domains domains
    # ========
    antecedent  = list(pattern.keys())

    _domains_set = np.random.choice(
        list(pattern.keys()), replace=False, size=2
    )

    cand = { d: pattern[d] for d in antecedent}

    # ===
    # Find k records with partial match with pattern
    # ===
    match_df = utils_local.query_df(test_df,cand)
    # search in train if no patterns in test set
    if len(match_df) < int( pattern_cluster_size):
        match_df = utils_local.query_df(train_df,cand)

    # upper bound on cluster size
    ns = min(len(match_df), pattern_cluster_size)
    res_df = pd.DataFrame(match_df.sample(ns))

    for d, e in pattern.items():
        res_df[d] = e

    pattern_idx_str = "0020" + str(p_idx)
    res_df[id_col] = res_df[id_col].apply(
        utils_local.aux_modify_id,
        args=(pattern_idx_str,)
    )

    return res_df

# ========== #

def generate_anomalies_type_2(
        train_df,
        test_df,
        save_dir,
        id_col='PanjivaRecordID',
        company_domain=None,
        reqd_anom_perc=50,
        num_jobs=40,
        pattern_min_support=5,
        pattern_cluster_count=100
):
    # =====================
    # Over estimating a bit, so that overlaps can be compensated for
    # distributed_pattern_count * cluster_size = total anomaly count
    # =====================
    distributed_pattern_count = int(1.10 * len(test_df) * (reqd_anom_perc / 100) / pattern_cluster_count )
    distributed_pattern_count = int(distributed_pattern_count/num_jobs)
    print('>>>', len(test_df))
    print('>>>', distributed_pattern_count)

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
            reqd_pattern_count=distributed_pattern_count,
            company_domain=company_domain
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

    print(' Percentage of anomalies :: ', len(patterns)*pattern_cluster_count / len(test_df) * 100)
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
            pattern_cluster_count
        ) for p_idx, pattern in zip(pattern_idx, patterns)
    )

    # Join the results
    anomalies_df = None
    for _df in res_df_list:
        if anomalies_df is None:
            anomalies_df = pd.DataFrame(_df, copy=True)
        else:
            anomalies_df = anomalies_df.append(_df, ignore_index=True)

    print(len(anomalies_df), len(test_df))
    op_path = os.path.join(save_dir, 'anomalies_type2.csv')
    anomalies_df.to_csv(op_path, index=None)
    return anomalies_df

# ========================
