import pandas as pd
import os
import numpy as np
from pandarallel import pandarallel
from joblib import Parallel, delayed

pandarallel.initialize(progress_bar=True, verbose=1)

from vose_sampler import VoseAlias
import pickle
import math
from itertools import combinations
from hashlib import md5
from multiprocessing import Pool
import multiprocessing

# ---- Global object ----- #
NODE_OBJECT_DICT = None
Serial_mapping_df = None
NO_REFRESH = True

# ------------------------- #

class Entity_Node:
    def __init__(self, domain=None, entity=None, serial_id=None):
        self.domain = domain
        self.entity = entity
        self.serial_id = serial_id
        self.transition_dict = {}
        self.nbr_types = []
        self.negative_nbr_dict = {}
        return

    def set_neighbor_types(self, list_nbr_types):
        self.nbr_types = list_nbr_types
        return

    # ----------------
    # Add transition probabilities to neighbors
    # Input :
    # nbr_type : domain name
    # un normalized counts_dict
    # ----------------
    def set_transition_prob(
            self,
            nbr_type,
            unnorm_counts_dict
    ):
        if nbr_type in self.transition_dict.keys():
            return

        self.nbr_types.append(nbr_type)
        unnorm_counts = list(unnorm_counts_dict.values())
        _sum = np.sum(unnorm_counts)
        norm_prob = [_ / _sum for _ in unnorm_counts]
        # -------------------
        # Dampen out rare values through raising them to power 0.75
        # -------------------
        p = [math.pow(_ / max(norm_prob), 0.75) for _ in norm_prob]
        p = [_ / sum(p) for _ in p]

        # prob_dist = {e[0]: e[1] for e in enumerate(p)}
        # set up the prob distribution dictionary as { key=serialized_id  : value=prob}

        id_list = list(unnorm_counts_dict.keys())
        prob_dist = {k: v for k, v in zip(id_list, p)}
        VA = VoseAlias(prob_dist)
        self.transition_dict[nbr_type] = VA
        return

    def set_negative_neighbor(
            self,
            nbr_type,
            unnorm_counts_dict
    ):
        if nbr_type in self.negative_nbr_dict.keys():
            return
        unnorm_counts = list(unnorm_counts_dict.values())
        # Sample negative neighbors uniformly
        t = np.logical_xor(unnorm_counts, np.ones(len(unnorm_counts), dtype=np.float))
        t = t.astype(np.float)
        t = t / sum(t)
        id_list = list(unnorm_counts_dict.keys())
        prob_dist = {k: v for k, v in zip(id_list, t)}

        VA = VoseAlias(prob_dist)
        self.negative_nbr_dict[nbr_type] = VA
        return

    def sample_nbr(self, nbr_type):
        if nbr_type not in self.transition_dict.keys():
            print('Neighbor type not in transition dictionary!!')
            print(self.transition_dict.keys())
            print(self.domain, 'Nbr type', nbr_type)
            print(self.entity)
            return None
        return self.transition_dict[nbr_type].sample_n(size=1)[0]

    def sample_negative_nbr(self, nbr_type):
        if nbr_type not in self.nbr_types:
            return None
        return self.negative_nbr_dict[nbr_type].sample_n(size=1)[0]

    def sample_multiple_negative_nbr(self, nbr_type, count):
        if nbr_type not in self.transition_dict.keys():
            return None
        return self.negative_nbr_dict[nbr_type].sample_n(size=count)


# ----------------------------------------------------------------- #

def get_coOccMatrixkey(a, b):
    if a < b:
        return '_+_'.join([a, b])
    else:
        return '_+_'.join([b, a])


class RandomWalkGraph_v1:
    Serial_2_Entity_ID_dict = {}
    Entity_2_Serial_ID_dict = {}
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
        global NO_REFRESH

        print('File :: ', self.node_obj_dict_file)

        if NO_REFRESH and os.path.exists(self.node_obj_dict_file):
            with open(self.node_obj_dict_file, "rb") as fh:
                self.node_object_dict = pickle.load(fh)
                return 1
        else:
            self.node_object_dict = {}
            # ------------------
            # Create node Objects for only node types in the path
            # ------------------
            for _domain_name in domain_dims.keys():
                self.node_object_dict[_domain_name] = {}
                for _id in range(domain_dims[_domain_name]):
                    _obj = Entity_Node(
                        domain=_domain_name,
                        entity=_id,
                        serial_id = RandomWalkGraph_v1.Serial_ID_lookup(_domain_name, _id)
                    )
                    self.node_object_dict[_domain_name][_id] = _obj

            return 2

    # ---------------------------------------------- #
    # Initialize the model
    # ---------------------------------------------- #

    def initialize(
            self,
            coOccMatrixDict,
            serial_mapping_df,
            domain_dims,
            id_col='PanjivaRecordID',
            MP_list = None,
            save_data_dir = None,
            random_walk_save_subdir = 'RW_Samples',
            saved_file_name='node_obj_dict.pkl'
    ):
        global NODE_OBJECT_DICT
        global Serial_mapping_df

        self.n_jobs = multiprocessing.cpu_count()
        self.MP_list = MP_list
        self.coOCcMatrix_dict = coOccMatrixDict
        self.domain_dims = domain_dims
        self.id_col = id_col
        self.save_data_dir = save_data_dir
        self.saved_file_name = saved_file_name
        signature = ''.join(
            ([''.join(_) for _ in MP_list])
        )
        self.random_walk_save_dir = os.path.join(
            self.save_data_dir,
            random_walk_save_subdir
        )

        if not os.path.exists(self.random_walk_save_dir):
            os.mkdir(self.random_walk_save_dir)


        self.signature = str(md5(str.encode(signature)).hexdigest())
        self.saved_file_name = saved_file_name.replace(
            '.',
            '_' + self.signature + '.'
        )
        Serial_mapping_df = serial_mapping_df
        RandomWalkGraph_v1.setup_Entity_ID_lookup()

        self.node_obj_dict_file = os.path.join(
            self.save_data_dir,
            self.saved_file_name
        )

        # ---------------------------------------- #
        # Check if already pre-computed dictionary and transitions.
        # ---------------------------------------- #
        # if os.path.exists(self.node_obj_dict_file):
        #     print('Node dict file exists :: ', self.saved_file_name)
        #     with open(self.node_obj_dict_file, "rb") as fh:
        #         self.node_object_dict = pickle.load(fh)
        #         NODE_OBJECT_DICT = self.node_object_dict
        #
        #     return
        # ----------------------------------------- #

        return_value = self.get_node_obj_dict(domain_dims)
        NODE_OBJECT_DICT = self.node_object_dict

        if return_value == 1 : return

        # ----------------------------------------- #
        # set up transition probabilities
        # ----------------------------------------- #
        def aux_f(
                obj,  # node_obj
                nbr_type,
                e_idx,
                orientation
        ):
            global Serial_mapping_df
            if orientation == 'r':
                arr = matrix[e_idx, :]
            else:
                arr = matrix[:, e_idx]

            # -----------------------------
            # Find the Serialized ids of the neighbors
            # -----------------------------
            _tmp_df = Serial_mapping_df.loc[
                Serial_mapping_df['Domain'] == nbr_type
                ].reset_index(drop=True)
            _tmp_df = _tmp_df.sort_values(by=['Entity_ID'])
            _tmp_df['count'] = arr

            _count_dict = { k: v for k, v in zip(
                list(_tmp_df['Serial_ID']), list(_tmp_df['count']))
            }

            obj.set_transition_prob(
                nbr_type=nbr_type,
                unnorm_counts_dict=_count_dict
            )
            obj.set_negative_neighbor(
                nbr_type=nbr_type,
                unnorm_counts_dict=_count_dict
            )
            return (e_idx, obj)

        # ------------------
        # Binary relations as per metapaths
        # ------------------
        relations = []
        for mp in MP_list:
            for _1, _2 in zip(mp[:-1], mp[1:]):
                relations.append('_+_'.join(sorted([_1, _2])))
        relations = set(relations)
        relations = [_.split('_+_') for _ in relations]
        print('Number of distinct relations :: ', len(relations) )

        for R in relations:
            print(' Relation :: ', R)
            domain_i = R[0]
            domain_j = R[1]
            # --------
            # Swap i,j so that i is lexicographically smaller than j
            # --------
            if domain_i > domain_j:
                (domain_i, domain_j) = (domain_j, domain_i)

            key = get_coOccMatrixkey(domain_i, domain_j)
            matrix = np.array(self.coOCcMatrix_dict[key])
            # ------------------------------
            # Consider both directions
            # i -> j  and j -> i
            # ------------------------------
            entities_i = [_ for _ in range(domain_dims[domain_i])]

            tmp = {idx: self.node_object_dict[domain_i][idx] for idx in entities_i}
            nbr_type = domain_j
            orientation = 'r'

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(aux_f)
                (obj,
                 nbr_type,
                 idx,
                 orientation, )
                for idx, obj in tmp.items()
            )

            for _res in results:
                e_i = _res[0]
                obj_i = _res[1]
                self.node_object_dict[domain_i][e_i] = obj_i

            entities_j = [_ for _ in range(domain_dims[domain_j])]
            tmp = {idx: self.node_object_dict[domain_j][idx] for idx in entities_j}
            nbr_type = domain_i
            orientation = 'c'

            res = Parallel(n_jobs=self.n_jobs)(
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
                self.node_object_dict[domain_j][e_j] = obj_j

        self.update_node_obj_dict()
        NODE_OBJECT_DICT = self.node_object_dict
        return

    @staticmethod
    def setup_Entity_ID_lookup():

        global Serial_mapping_df
        RandomWalkGraph_v1.Serial_2_Entity_ID_dict = {}
        RandomWalkGraph_v1.Entity_2_Serial_ID_dict = {}

        for domain in set(Serial_mapping_df['Domain']):

            RandomWalkGraph_v1.Serial_2_Entity_ID_dict[domain] = {}
            tmp = Serial_mapping_df.loc[
                (Serial_mapping_df['Domain'] == domain)]
            s_id_list = list(tmp['Serial_ID'])
            e_id_list = list(tmp['Entity_ID'])
            for _si, _ej in zip(s_id_list, e_id_list):
                RandomWalkGraph_v1.Serial_2_Entity_ID_dict[domain][_si] = _ej

            RandomWalkGraph_v1.Entity_2_Serial_ID_dict[domain] = {}
            for _si, _ej in zip(s_id_list, e_id_list):
                RandomWalkGraph_v1.Entity_2_Serial_ID_dict[domain][_ej] = _si

    @staticmethod
    def Entity_ID_lookup(domain, serial_id):
        return RandomWalkGraph_v1.Serial_2_Entity_ID_dict[domain][serial_id]

    @staticmethod
    def Serial_ID_lookup(domain,entity_id):
        return RandomWalkGraph_v1.Entity_2_Serial_ID_dict[domain][entity_id]

    # ----------------------
    # Takes 4 arguments:
    # 1. entity id of starting node, belonging to domain of 1st type in metapath
    # 2. actual meta path (symmetric)
    # 3. rw_count
    # 4. number of negative samples
    # ----------------------
    @staticmethod
    def aux_rw_exec_w_ns(
            args
    ):
        global NODE_OBJECT_DICT
        global Serial_mapping_df

        print('In aux_rw_exec_w_ns')
        print(args)
        start_node_entity_idx = args[0]
        mp = args[1]
        rw_count = args[2]
        rw_length = args[3]

        num_neg_samples = args[4]
        node_object_dict = NODE_OBJECT_DICT
        augmented_mp = mp + mp[::-1][1:]

        rep_len = len(augmented_mp) - 1
        path_seq = augmented_mp + augmented_mp[1:] * (rw_length // rep_len)
        print('Augmented MP ', path_seq)

        # --------------
        # Pad it at  end
        # --------------
        path_seq = path_seq[:rw_length + 1]
        all_walks = []
        all_neg_samples = []
        # --------------------------
        # Note :: ensure no cycles
        # --------------------------
        for rc in range(rw_count):
            cycle_prevention_dict = {_: [] for _ in mp}
            cur_domain = mp[0]
            _obj = node_object_dict[cur_domain][start_node_entity_idx]
            print( _obj,  _obj.entity, _obj.domain )

            cur_node_s_id = RandomWalkGraph_v1.Serial_ID_lookup(cur_domain, start_node_entity_idx)
            # cur_node_s_id = node_object_dict[cur_domain][start_node_entity_idx].serial_id
            cycle_prevention_dict[cur_domain].append(cur_node_s_id)

            neg_samples = None
            walk = []
            nbr_e_id = None

            for i in range(0, rw_length):
                if i == 0:
                    cur_node_e_id = start_node_entity_idx
                    cur_domain = mp[0]
                    cur_node_s_id = node_object_dict[cur_domain][cur_node_e_id].serial_id
                    cur_node_s_id = RandomWalkGraph_v1.Serial_ID_lookup(cur_domain, cur_node_e_id)
                else:
                    cur_domain = path_seq[i]
                    # ------
                    # next neighbor determined in previous step
                    # nbr_e_id is entity id of a current domain selected in previous step
                    # that becomes cur node
                    # ------
                    cur_node_e_id = nbr_e_id
                    cur_node_s_id = node_object_dict[cur_domain][cur_node_e_id].serial_id
                    cur_node_s_id = RandomWalkGraph_v1.Serial_ID_lookup(cur_domain, cur_node_e_id)

                cur_node_obj = node_object_dict[cur_domain][cur_node_e_id]

                # Add the current id to walk
                walk.append(cur_node_s_id)
                # For the next step
                next_nbr_domain = path_seq[i + 1]

                nbr_s_id = cur_node_obj.sample_nbr(next_nbr_domain)
                print(next_nbr_domain,nbr_s_id)
                # while  nbr_s_id in cycle_prevention_dict[next_nbr_domain]:
                #     nbr_s_id = cur_node_obj.sample_nbr(next_nbr_domain)

                cycle_prevention_dict[next_nbr_domain].append(nbr_s_id)

                if nbr_s_id is None:
                    print('Error!!')
                    return None

                nbr_e_id = RandomWalkGraph_v1.Entity_ID_lookup(
                    next_nbr_domain,
                    nbr_s_id
                )

                # ----- Get the negative samples -----
                # Negative samples are nodes to which a random walk would not happen from current node
                # -------------------------------------

                # Go to next node and sample nodes of current domain

                ns_next_nbr_obj = node_object_dict[next_nbr_domain][nbr_e_id]
                _neg_samples = ns_next_nbr_obj.sample_multiple_negative_nbr(
                    cur_domain,
                    num_neg_samples
                )

                # _neg_samples has shape [ ?, ns ]
                _neg_samples = np.reshape(_neg_samples, [-1, 1])

                if neg_samples is None:
                    neg_samples = _neg_samples
                else:
                    neg_samples = np.hstack([neg_samples, _neg_samples])

                # ---------------------------------- #

            # -------------------------------------- #
            # remove the last one ; since padding was done
            # -------------------------------------- #

            all_walks.append(walk)
            all_neg_samples.append(neg_samples)

        all_neg_samples = np.stack(all_neg_samples, axis=0)
        return (all_walks, all_neg_samples)

    # ------------------------------------
    # Function to generate Random Walks  and
    # Negative samples at each point to serve as context
    # ------------------------------------
    def generate_RandomWalks_w_neg_samples(
            self,
            mp = None,
            rw_count=250,
            rw_length=100,
            num_neg_samples=10
    ):

        MP_list = []
        if mp is not None:
            # check if valid
            tmp = []
            for _mp in self.MP_list:
                tmp.append('_'.join(_mp))
            _c = '_'.join(mp)
            if _c in tmp:
                MP_list = [mp]
        else:
            MP_list = list(self.MP_list)

        num_jobs = max(8, self.n_jobs)
        print('[INFO] Meta paths ', self.MP_list)
        print('[INFO] Number of jobs ', num_jobs)

        #  =================
        # Do RW for each of the domain entities in the meta path
        #  =================
        for _MP in MP_list:
            meta_path_pattern = _MP
            print('Path :: ', meta_path_pattern)
            # Start the random walk from start node
            domain_t = _MP[0]
            start_nodes_idx = []
            for e_id in range(self.domain_dims[domain_t]):
                start_nodes_idx.append(e_id)

            result = None
            neg_samples = []
            args = [
                (n, meta_path_pattern, rw_count, rw_length, num_neg_samples)
                for n in start_nodes_idx
            ]

            with Pool(num_jobs) as p:
                pooled_result = p.map(
                    RandomWalkGraph_v1.aux_rw_exec_w_ns,
                    args
                )

            for res in pooled_result:
                walk = res[0]
                neg_samples.append(res[1])

                if result is None:
                    result = walk
                else:
                    result.extend(walk)

            # Save the Random Walks as numpy array
            fname = '_'.join(_MP) + '_walks.npy'
            fpath = os.path.join(self.random_walk_save_dir, fname)
            result = np.array(result)
            np.save(fpath, result)

            # ---- Save the negative samples as a numpy array ------ #
            fname = '_'.join(_MP) + '_neg_samples.npy'
            fpath = os.path.join(self.random_walk_save_dir, fname)
            neg_samples = np.concatenate(neg_samples)
            np.save(fpath, neg_samples)

        return
