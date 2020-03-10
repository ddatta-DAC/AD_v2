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
from multiprocessing import Pool
import multiprocessing

# ---- Global object ----- #
NODE_OBJECT_DICT = None
# ------------------------- #

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

    def sample_nbr(self, nbr_type):
        if nbr_type not in self.nbr_types:
            return None
        return self.transition_dict[nbr_type].sample_n(size=1)[0]
# ----------------------------------------------------------------- #

def get_key(a, b):
    if a < b:
        return '_+_'.join([a, b])
    else:
        return '_+_'.join([b, a])


class RandomWalkGraph_v1:
    def __init__(self):
        self.df_x = None
        self.MP_list = []
        self.node_obj_dict_file = None
        self.save_data_dir = None
        self.node_object_dict = {}
        return


    def update_node_obj_dict(
        self
    ):
        print('Saving file : ', self.node_obj_dict_file)
        with open(self.node_obj_dict_file, "wb") as fh:
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
        print(self.node_obj_dict_file)
        NO_REFRESH = True
        if NO_REFRESH and os.path.exists(self.node_obj_dict_file):
            with open(self.node_obj_dict_file, "rb") as fh:
                self.node_object_dict = pickle.load(fh)
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
                    self.node_object_dict[_domain_name][_id] = _obj
            self.update_node_obj_dict()

        return


    def get_coOccMatrixDict(
            self,
            df_x
        ):
        coOccMatrix_File = os.path.join(self.save_data_dir, 'coOccMatrixSaved.pkl')
        if not os.path.exists(coOccMatrix_File):
            coOCcMatrix_dict = cMg.get_coOccMatrix_dict(df_x, self.id_col)
            with open(coOccMatrix_File, 'wb') as fh:
                pickle.dump(coOCcMatrix_dict, fh, pickle.HIGHEST_PROTOCOL)
        else:
            with open(coOccMatrix_File, 'rb') as fh:
                coOCcMatrix_dict = pickle.load(fh)
        return coOCcMatrix_dict

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
        global NODE_OBJECT_DICT
        self.MP_list = MP_list
        self.domain_dims = domain_dims
        self.id_col = id_col
        self.save_data_dir = save_data_dir
        self.saved_file_name = saved_file_name
        _signature = ''.join(sorted([''.join(_) for _ in MP_list]))

        self.signature = str(md5(str.encode(_signature)).hexdigest())
        self.saved_file_name = saved_file_name.replace(
            '.',
            '_' + self.signature + '.'
        )

        self.node_obj_dict_file = os.path.join(
            self.save_data_dir,
            self.saved_file_name
        )
        print(' >> ', self.saved_file_name)

        if os.path.exists(self.node_obj_dict_file):
            print( 'Node dict file exists !!')
            with open(self.node_obj_dict_file,"rb") as fh:
                self.node_object_dict = pickle.load(fh)
                NODE_OBJECT_DICT = self.node_object_dict
            return

        # ----------------------------------------- #
        self.coOCcMatrix_dict = self.get_coOccMatrixDict(df_x)
        self.get_node_obj_dict(domain_dims )
        NODE_OBJECT_DICT = self.node_object_dict

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
            matrix = np.array(self.coOCcMatrix_dict[key])
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

    @staticmethod
    def aux_rw_exec_1(
            args
    ):
        global NODE_OBJECT_DICT

        start_node_idx = args[0]
        domain_steps = args[1]
        rw_count = args[2]
        node_object_dict = NODE_OBJECT_DICT

        all_walks = []
        for i in range(rw_count):
            _domain_steps = list(domain_steps)
            cur_node_idx = start_node_idx
            walk_idx = []
            while len(_domain_steps) > 0:
                walk_idx.append(cur_node_idx)
                cur_domain = _domain_steps.pop(0)
                cur_node = node_object_dict[cur_domain][cur_node_idx]
                if len(_domain_steps) == 0: break
                nxt_domain = _domain_steps[0]
                nxt_e_idx = cur_node.sample_nbr(nxt_domain)
                cur_node_idx = nxt_e_idx
            all_walks.append(walk_idx)
        return all_walks

    # ---------------------------------- #
    # Function to get the random walks
    # ---------------------------------- #
    def generate_RandomWalks(
            self,
            mp=None,
            rw_count = 5
    ):
        if mp is not None:
            # check if valid
            tmp =[]
            for _mp in self.MP_list:
                tmp.append('_'.join(_mp))
            _c = '_'.join(mp)
            if _c in tmp :
                MP_list = [mp]
        else:
            MP_list = list(self.MP_list)

        print(' Keys node_object_dict', self.node_object_dict.keys())
        print('Meta paths', self.MP_list)
        num_jobs = max(4, multiprocessing.cpu_count())
        print('Number of jobs ', num_jobs)
        _dir = os.path.join(self.save_data_dir, 'RW_Samples')

        if not os.path.exists(_dir):
            os.mkdir(_dir)

        for _MP in MP_list:
            # Do RW for each of the domain entities in the meta path
            path_queue = _MP + _MP[::-1][1:]
            print(path_queue)

            # Start the random walk from start node
            domain_t = _MP[0]
            start_nodes_idx = []

            for e_id in range(self.domain_dims[domain_t]):
                start_nodes_idx.append(e_id)

            res = None
            tmp_res =[]
            with Pool(num_jobs) as p:
                args = [
                    ( n ,path_queue, rw_count)
                    for n in start_nodes_idx
                ]
                tmp = p.map(
                    RandomWalkGraph_v1.aux_rw_exec_1,
                    args
                )
                tmp_res.extend(tmp)

            for _r in tmp_res :
                tmp = _r
                if res is None: res = tmp
                else: res.extend(tmp)

            df = pd.DataFrame(res, columns=path_queue)
            # Save DataFrame
            fname = '_'.join(_MP) + '.csv'
            fpath = os.path.join(
                _dir,
                fname
            )
            print(fpath)
            df.to_csv(fpath,index=None)
        return













