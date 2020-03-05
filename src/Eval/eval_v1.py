from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import numpy as np


def precision_recall_curve(
        sorted_id_score_dict,
        anomaly_id_list,
        bounds = None
):
    recall = 0
    correct = 0
    recall_vals = []
    precision_vals = []
    num_anomalies = len(anomaly_id_list)
    print('Number of Anomalies ', num_anomalies)
    total_count = len(sorted_id_score_dict)

    '''
    Following charu aggarwal
        Precision  = (S(t) intersection G) / |S(t)|
        Recall  = (S(t) intersection G) / |G|
    '''
    if bounds  is None :
        min_val = min( sorted_id_score_dict.values())
        max_val = max( sorted_id_score_dict.values())
    else:
        min_val = bounds[0]
        max_val = bounds[1]

    step = (max_val-min_val)/100
    prev_cand_count = 0

    for t in np.arange(min_val, max_val + step, step):
        # find number of records marked anomalies at this threshold
        candidates = [_ for _,val in sorted_id_score_dict.items() if val <= t ]
        if prev_cand_count == len(candidates) :
            continue
        prev_cand_count = len(candidates)

        _numerator = len(
            set(candidates).intersection(anomaly_id_list)
        )

        p = _numerator/len(candidates)
        r = _numerator/num_anomalies

        precision_vals.append(p)
        recall_vals.append(r)
        if r == 1:
            break

        # -------------------------- #
    return recall_vals, precision_vals
