import pandas as pd
import os
import numpy as np
from pandarallel import pandarallel
from joblib import Parallel, delayed
pandarallel.initialize(progress_bar=True, verbose=1)
from src.utils import coOccMatrixGenerator as cMg
from vose_sampler import VoseAlias
import pickle
import math
from itertools import combinations
from hashlib import md5

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
        if nbr_type in self.transition_dict.keys():
            return

        self.nbr_types.append(nbr_type)
        _sum = np.sum(unnorm_counts)
        norm_prob = [_ / _sum for _ in unnorm_counts]
        # -------------------
        # Dampen out rare values through raising them to power 0.75
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
# ----------------------------------------------------------------- #

def get_key(a, b):
    if a < b:
        return '_+_'.join([a, b])
    else:
        return '_+_'.join([b, a])


class RandomWalker_v1:
    def __init__(self):
        self.df_x = None
        self.MP_list = []
        self.node_obj_dict_file = None
        self.save_data_dir = None
        self.node_object_dict = None
        return


    def update_node_obj_dict(
        self
    ):
        with open(self.node_object_dict_file, "wb") as fh:
            pickle.dump(
                self.node_object_dict,
                fh,
                pickle.HIGHEST_PROTOCOL
            )
        return

    # ------------------------
    # node_obj_dict
    # { domain : { 0 : <obj>, 1 :<obj>, 2 : <obj>, ... }, ... }
    # ------------------------
    def get_node_obj_dict(
            self,
            domain_dims
    ):

        NO_REFRESH = False
        if NO_REFRESH and os.path.exists(self.node_obj_dict_file):
            with open(self.node_obj_dict_file, "rb") as fh:
                self.node_object_dict  = pickle.load(fh)
        else:
            self.node_object_dict  = {}
            # Create node objects for only node types in the path
            for _domain_name in domain_dims.keys():
                self.node_object_dict [_domain_name] = {}
                for _id in range(domain_dims[_domain_name]):
                    _obj = Entity_Node(
                        domain=_domain_name,
                        entity=_id
                    )
                    self.node_object_dict [_domain_name][_id] = _obj
        return


    def get_coOccMatrixDict(
            self,
            df_x,
            save_data_dir,
            id_col
        ):
        coOccMatrix_File = os.path.join(save_data_dir, 'coOccMatrixSaved.pkl')
        if not os.path.exists(coOccMatrix_File):
            coOCcMatrix_dict = cMg.get_coOccMatrix_dict(df_x, id_col)
            with open(coOccMatrix_File, 'wb') as fh:
                pickle.dump(coOCcMatrix_dict, fh, pickle.HIGHEST_PROTOCOL)
        else:
            with open(coOccMatrix_File, 'rb') as fh:
                coOCcMatrix_dict = pickle.load(fh)
        return coOCcMatrix_dict

    # def setup_MP(
    #     self,
    #     domain_dims,
    #     meta_path_seq=[],
    #     symmetric=True,
    #     save_data_dir=None,
    #     id_col='PanjivaRecordID',
    # ):
    #     if len(meta_path_seq) < 2:
    #         print('Error')
    #         exit(1)
    #     if symmetric:
    #         MP = meta_path_seq + meta_path_seq[::-1][1:]
    #     else:
    #         MP = meta_path_seq
    #     return


# ---------------------------------------------- #

    def initialize(
            self,
            df_x,
            domain_dims,
            id_col='PanjivaRecordID',
            MP_list = [],
            save_data_dir = None,
            saved_file_name = 'node_obj_dict.pkl'
    ):
        self.save_data_dir = save_data_dir
        self.saved_file_name = saved_file_name
        _signature = ''.join(sorted([''.join(_) for _ in MP_list]))
        self.signature = str(md5(str.encode(_signature)).hexdigest())

        self.saved_file_name = saved_file_name.replace(
            '.',
            '_'+ self.signature + '.'
        )

        self.node_object_dict_file = os.path.join(
            self.save_data_dir,
            self.saved_file_name
        )
        print(self.saved_file_name)
        if os.path.exists(self.node_object_dict_file):
            with open(self.node_object_dict_file,"rb") as fh:
                self.node_object_dict_file = pickle.load(fh)
            return

        self.node_object_dict = {}
        coOCcMatrix_dict = self.get_coOccMatrixDict(df_x, save_data_dir, id_col)
        self.get_node_obj_dict(domain_dims )

        # ----------------------------------------- #
        # set up transition probabilities
        # ----------------------------------------- #
        def aux_f(
                obj,
                nbr_type,
                idx,
                orientation
        ):
            if orientation == 'r':
                arr = matrix[idx, :]
            else:
                arr = matrix[:, idx]

            obj.set_transition_prob(
                nbr_type = nbr_type,
                unnorm_counts = arr
            )
            return (idx, obj)

        relations = []
        for mp in MP_list:
            for _1, _2 in zip( mp[:-1],mp[1:]):
                relations.append([_1,_2])
            print(' >> ',relations)

        for R in relations:
            print(' Relation :: ', R)
            i = R[0]
            j = R[1]
            # --------
            # Swap i,j so that i is lexicographically smaller than j
            # --------
            if i > j:
                (i, j) = (j, i)

            key = get_key(i, j)
            matrix = np.array(coOCcMatrix_dict[key])
            # ------------------------------
            # Consider both directions
            # i -> j  and j -> i
            # ------------------------------
            entities_i = [_ for _ in range(domain_dims[i])]
            tmp = { idx : self.node_object_dict[i][idx]  for idx in  entities_i}
            nbr_type = j
            orientation = 'r'

            res = Parallel(n_jobs=8)(
                delayed(aux_f)
                (obj,
                nbr_type,
                idx,
                orientation)
                for idx,obj in tmp.items()
            )

            for _r in res:
                e_i =_r[0]
                obj_i = _r[1]
                self.node_object_dict[i][e_i] = obj_i

            entities_j = [_ for _ in range(domain_dims[j])]
            tmp = {idx:  self.node_object_dict[j][idx] for idx in entities_j}
            nbr_type = i
            orientation = 'c'

            res = Parallel(n_jobs=8)(
                delayed(aux_f)
                (obj,
                 nbr_type,
                 idx,
                 orientation)
                for idx, obj in tmp.items()
            )

            for _r in res:
                e_j = _r[0]
                obj_j = _r[1]
                self.node_object_dict[j][e_j] = obj_j

        self.update_node_obj_dict()
        return


