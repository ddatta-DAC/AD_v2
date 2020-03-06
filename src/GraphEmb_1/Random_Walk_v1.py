import pandas as pd
import os
import numpy as np
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, verbose=1)
from src.utils import coOccMatrixGenerator as cMg
from vose_sampler import VoseAlias
import pickle
import math


class Entity_Node:
    def __init__(self, domain, entity):
        self.domain = domain
        self.entity = entity
        self.transition_dict = {}
        self.nbr_types = []
        return

    def set_neighbor_types(self, list_nbr_types):
        self.nbr_types = list_nbr_types
        return

    def set_transition_prob(
            self,
            nbr_type,
            unnorm_counts
    ):
        self.nbr_types.append(nbr_type)
        _sum = np.sum(unnorm_counts)
        norm_prob = [_ / _sum for _ in unnorm_counts]

        # -------------------
        # Dampen out rare values through raisuingh them to power 0.75
        # -------------------
        p = [math.pow(_ / max(norm_prob), 0.75) for _ in norm_prob]
        p = [_ / sum(p) for _ in p]
        prob_dist = {e[0]: e[1] for e in enumerate(p)}

        VA = VoseAlias(prob_dist)
        self.transition_dict[nbr_type] = VA
        return

    def sample(self, nbr_type):
        if nbr_type not in self.nbr_types:
            return None
        return self.transition_dict[nbr_type].sample_n(size=1)[0]


# ------------------------
# node_obj_dict
# { domain : { 0 : <obj>, 1 :<obj>, 2 : <obj>, ... }, ... }
# ------------------------
def get_node_obj_dict(
        domain_dims,
        saved_data_dir=None
):
    node_obj_dict_file = os.path.join(
        saved_data_dir,
        'node_obj_dict.pkl'
    )
    if os.path.exists(node_obj_dict_file):
        with open(node_obj_dict_file, "rb") as fh:
            node_object_dict = pickle.load(fh)
    else:
        node_object_dict = {}
        # Create node objects for only node types in the path
        for _domain_name in domain_dims.keys():
            node_object_dict[_domain_name] = {}
            for _id in range(domain_dims[_domain_name]):
                _obj = Entity_Node(
                    domain=_domain_name,
                    entity=_id
                )
                node_object_dict[_domain_name][_id] = _obj

    return node_object_dict


def generate_all_rw(
        df_x,
        domain_dims,
        meta_path_seq=[],
        symmetric=True,
        id_col='PanjivaRecordID',
        saved_data_dir=None
):
    if len(meta_path_seq) < 2:
        print('Error')
        exit(1)

    if symmetric:
        MP = meta_path_seq + meta_path_seq[::-1][1:]
    else:
        MP = meta_path_seq
    path_id = 0

    coOccMatrix_File = os.path.join(saved_data_dir, 'coOccMatrixSaved.pkl')
    if not os.path.exists(coOccMatrix_File):
        coOCcMatrix_dict = cMg.get_coOccMatrix_dict(df_x, id_col)
        with open(coOccMatrix_File, 'wb') as fh:
            pickle.dump(coOCcMatrix_dict, fh, pickle.HIGHEST_PROTOCOL)
    else:
        with open(coOccMatrix_File, 'rb') as fh:
            coOCcMatrix_dict = pickle.load(fh)

    # set up transition probabilities
    relations = []
    for i, j in zip(MP[:-1], MP[1:]):
        relations.append(list([i, j]))

    node_object_dict = get_node_obj_dict(domain_dims, saved_data_dir)

    def get_key(a, b):
        if a < b:
            return '_+_'.join([a, b])
        else:
            return '_+_'.join([b, a])

    for R in relations:
        i = R[0]
        j = R[1]

        # --------
        # Swap i,j so that i is lexicographically smaller than j
        # --------
        if i > j:
            (i, j) = (j, i)

        key = get_key(i, j)
        matrix = coOCcMatrix_dict[key]

        # Consider both directions
        # i -> j  and j -> i

        entities_i = [_ for _ in range(domain_dims[i])]
        for e_i in entities_i:
            obj = node_object_dict[i][e_i]
            obj.set_transition_prob(
                nbr_type=j,
                unnorm_counts=matrix[i, :]
            )

        entities_j = [_ for _ in range(domain_dims[j])]
        for e_i in entities_j:
            obj = node_object_dict[i][e_i]
            obj.set_transition_prob(
                nbr_type=j,
                unnorm_counts=matrix[:, j]
            )

    return
