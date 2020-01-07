import pandas as pd
import sys
import os
import numpy as np
import pickle
from itertools import combinations
from joblib import Parallel, delayed
from numpy import random
import hashlib
from collections import Counter

sys.path.append('./')

try:
    from . import utils_createAnomalies as utils_local
    from . import clean_up_test_data
except:
    import utils_createAnomalies as utils_local
    import clean_up_test_data


'''
Type 3 :
Select one entity as a "company" (shipper or consignee) which has probability  < median(probability)
Select two more entities such that they do not co-occur with the company, in the training set 

# =====
Algo :
Select nc = k/p company entities , where p are number of company domains
k =  A / c 
c = "cluster" size
A = count of anomalies needed, determined by anom_percentage variable;  ap /100 * len(test data)
Dispatch method for generating c anomalies  
:: using each {company domain, company entity entity} pair (1,2 ... nc)

'''

def generate_anomalies_type3_aux2(
        test_df,
        train_df,
        p_idx,
        id_col,
        fixed_domain,
        entity_sample_fd,
        domain_entitiesSet_dict,
        columnPair_coOccMatrix_dict,
        cluster_count,

):
    domainEntity_dict = {fixed_domain: entity_sample_fd}
    match_df = utils_local.query_df(
        test_df,
        domainEntity_dict
    )
    # We want duplicates of anomalous patterns
    match_df.sample(cluster_count)

    # Select 2 entities that do not co-occur with this entity(company)
    possible_domains = list(domain_entitiesSet_dict.keys())
    possible_domains.remove(fixed_domain)
    sampled_domains = np.random.sample(
        possible_domains,
        size=2,
        replace=False
    )

    # This DF holds the result of the search
    new_df = pd.DataFrame(columns=list(train_df.columns))
    candidate_dict = {}
    for sd in sampled_domains:
        found = False
        while not found:
            se = np.random.sample(
                domain_entitiesSet_dict,
                size=1,
                replace=False
            )[0]
            tmp_dict = {
                sd: se,
                fixed_domain: entity_sample_fd
            }

            if utils_local.check_nonZeroCoOccurrence(tmp_dict, columnPair_coOccMatrix_dict) == False:
                candidate_dict[sd] = se
                break

    for _, row in match_df.iterrows():
        copy_row = pd.Series(
            row, copy=True
        )
        for _d, _e in candidate_dict.items():
            copy_row[_d] = _e
        new_df = new_df.append(copy_row, ignore_index=True)

    _pattern_identifier_str = '0030'+ str(p_idx)
    new_df[id_col] = new_df[id_col].apply(
        utils_local.aux_modify_id,
        args=(_pattern_identifier_str,)
    )
    return new_df



def generate_anomalies_type3_aux1(
    test_df,
    train_df,
    pattern_idx_start,
    id_col,
    fixed_domain,
    entity_samples,
    domain_entitiesSet_dict,
    columnPair_coOccMatrix_dict,
    num_jobs = 40,
    cluster_count = 100
    ):

    pattern_idx_list = list(range(pattern_idx_start, len(entity_samples)))

    list_res_df = Parallel(n_jobs=num_jobs)(
        delayed(generate_anomalies_type3_aux2)(
            test_df,
            train_df,
            p_idx,
            id_col,
            fixed_domain,
            entity_sample_fd,
            domain_entitiesSet_dict,
            columnPair_coOccMatrix_dict,
            cluster_count
        ) for p_idx,entity_sample_fd in zip(pattern_idx_list, entity_samples)
    )
    res_df = None
    for _df in list_res_df:
        if res_df is None:
            res_df = pd.DataFrame( _df, copy=True )
        else:
            res_df = res_df.append(_df, ignore_index=True)

    return res_df

# ========================================== #

def generate_anomalies_type3(
        test_df,
        train_df,
        save_dir,
        id_col='PanjivaRecordID',
        company_domains = None,
        num_jobs=40,
        anom_perc=10,
        cluster_count = 100
):
    domains = list(sorted(test_df.columns))
    domains.remove(id_col)

    # create set of entity ids for each of the domains
    domain_entitiesSet_dict = {}
    for d in domains:
        domain_entitiesSet_dict[d] = list(set(train_df[d]))
    # Create the co-occurrence matrix using the reference data frame(training data)
    columnPair_coOccMatrix_dict = utils_local.get_coOccMatrix_dict(train_df, id_col)

    company_domain_candidates = {}
    req_count = int(len(test_df) * (anom_perc / 100) / cluster_count * (1 / len(company_domains)))

    anomalies_df = None

    pattern_idx_start = 0
    for cd in company_domains:
        count_dict  = dict(Counter(train_df[cd]))
        # get the median
        _median = np.median( list(count_dict.values()) )
        candidates = [ c for e, c in count_dict.items() if c < _median ]

        # If there are more than one Company domain, split evenly
        company_domain_candidates[cd] = np.random.choice(
            candidates,
            size= req_count,
            replace=False
        )

        res_df = generate_anomalies_type3_aux1(
            test_df,
            train_df,
            pattern_idx_start,
            id_col=id_col,
            fixed_domain=cd,
            entity_samples=company_domain_candidates[cd],
            domain_entitiesSet_dict=domain_entitiesSet_dict,
            columnPair_coOccMatrix_dict=columnPair_coOccMatrix_dict,
            num_jobs=num_jobs,
            cluster_count=cluster_count
        )
        pattern_idx_start += len(company_domain_candidates[cd])
        if anomalies_df is None:
            anomalies_df = pd.DataFrame( res_df, copy=True)
        else:
            anomalies_df = anomalies_df.append(res_df,ignore_index=True)

    # ============

    op_path = os.path.join(save_dir, 'anomalies_type3.csv')
    anomalies_df.to_csv(op_path, index=None)
    return anomalies_df
