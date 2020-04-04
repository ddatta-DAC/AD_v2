#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------
import pickle
import sys
import os
import pandas as pd
import networkx as nx
from networkx import simrank_similarity_numpy
from collections import defaultdict
import multiprocessing

sys.path.append('./../..')
sys.path.append('./..')
from joblib import Parallel, delayed
from src.utils import coOccMatrixGenerator

try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher

nodeObj_dict_file = 'nodeObj_dict.pkl'
REFRESH_NODES = False
id_col = 'PanjivaRecordID'
nodeObj_Dict = None
model_use_data_DIR = None


# ------------------------------------------------------------3
# Call this to set up global variables
# ------------------------------------------------------------
def initialize(_dir, _model_use_data_DIR):
    global DIR
    global model_use_data_DIR

    DIR = _dir
    if _model_use_data_DIR is None:
        model_use_data_DIR = 'model_use_data'
    else:
        model_use_data_DIR = _model_use_data_DIR

    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    if not os.path.exists(os.path.join(model_use_data_DIR, DIR)):
        os.mkdir(os.path.join(model_use_data_DIR, DIR))
    model_use_data_DIR = os.path.join(model_use_data_DIR, DIR)
    return


# ------------------------------------------------------------
def get_training_data(DIR):
    SOURCE_DATA_DIR = './../../generated_data_v1'
    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR, DIR)
    return data


def get_domain_dims(DIR):
    with open(
            os.path.join(
                './../../generated_data_v1/',
                DIR,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)
    return domain_dims


# ------------------------------------
class node:
    id = 0

    @staticmethod
    def get_id():
        _id = node.id
        node.id = node.id + 1
        return _id

    def __init__(self, node_type, e_id):
        self.node_type = node_type
        self.id = node.get_id()
        self.e_id = e_id


# Create the nodes first

def create_transaction_nodes(list_record_ids):
    global id_col
    global nodeObj_Dict
    domain = id_col
    nodeObj_Dict[domain] = defaultdict()
    for e_id in sorted(list_record_ids):
        node_obj = node(node_type=domain, e_id=e_id)
        nodeObj_Dict[domain][e_id] = node_obj
    return


def create_entity_nodes(domain_dims, nodeObj_dict):
    for domain, size in domain_dims.items():
        nodeObj_dict[domain] = defaultdict()
        for e_id in range(size):
            node_obj = node(node_type=domain, e_id=e_id)
            nodeObj_dict[domain][e_id] = node_obj
    return nodeObj_dict


# ---------------------------
# Use the training data to create nodes
# Initially create entity nodes only
# ---------------------------
def get_base_nodeObj_dict(
        domain_dims,
        DIR
):
    global id_col
    global REFRESH_NODES
    global nodeObj_dict_file
    global model_use_data_DIR

    f_path = os.path.join(model_use_data_DIR, nodeObj_dict_file)
    print('Path :: ', f_path)
    if os.path.exists(f_path):
        with open(f_path, 'rb') as fh:
            nodeObj_dict = pickle.load(fh)
            return nodeObj_dict

    nodeObj_dict = defaultdict()
    nodeObj_dict = create_entity_nodes(domain_dims, nodeObj_dict)
    with open(f_path, 'wb') as fh:
        pickle.dump(
            nodeObj_dict,
            fh,
            pickle.HIGHEST_PROTOCOL
        )
    return nodeObj_dict


# ------------------------------
# Read in list of edge types
# ------------------------------
def read_schema():
    relations_list = []
    with open('schema.txt', 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            line = line.strip()
            _list = line.split(',')
            _list = [_.strip() for _ in _list]
            relations_list.append(_list)
    return relations_list


def get_coOcc_dict(
        train_x_df,
        id_col='PanjivaRecordID'
):
    global model_use_data_DIR
    coOccDict_file = os.path.join(model_use_data_DIR, 'coOccDict.pkl')
    if os.path.exists(coOccDict_file):
        with open(coOccDict_file, 'rb') as fh:
            coOccDict = pickle.load(fh)
    else:
        coOccDict = coOccMatrixGenerator.get_coOccMatrix_dict(train_x_df, id_col)
        with open(coOccDict_file, 'wb') as fh:
            pickle.dump(
                coOccDict,
                fh,
                pickle.HIGHEST_PROTOCOL
            )
    return coOccDict


def save_graph(G, graph_file='nx_G0.pkl'):
    global model_use_data_DIR

    graph_obj_path = os.path.join(model_use_data_DIR, graph_file)
    nx.write_gpickle(G, graph_obj_path, protocol=4)

    return


def read_graph(graph_file='nx_G1.pkl'):
    global model_use_data_DIR
    graph_file = 'nx_G1.pkl'
    graph_obj_path = os.path.join(model_use_data_DIR, graph_file)
    if os.path.exists(graph_obj_path):
        G = nx.read_gpickle(graph_obj_path)
        return G
    else:
        return None


# ----------------------------------------------------------- #
# Function to get initial graph (with nodes as entities)
# ----------------------------------------------------------- #
def get_initial_graph(
        df,
        domain_dims
):
    global nodeObj_Dict
    graph_file = 'nx_G0.pkl'

    G0 = read_graph(graph_file)
    if G0 is not None:
        return G0

    nodeObj_Dict = get_base_nodeObj_dict(domain_dims, DIR)
    print(domain_dims)

    list_edge_types = read_schema()
    coOccDict = get_coOcc_dict(df)

    G = nx.Graph()
    # Add in the nodes
    for domain, _dict in nodeObj_Dict.items():
        list_nodes = list(_dict.values())
        print(domain, len(list_nodes))
        G.add_nodes_from(list_nodes)

    # Add edges
    n_jobs = multiprocessing.cpu_count()
    from multiprocessing import Pool
    from itertools import product

    for edge in list_edge_types:
        if id_col in edge: continue
        # get co_occurrence_matrix
        domain_1 = edge[0]
        domain_2 = edge[1]
        if domain_1 > domain_2:
            domain_1, domain_2 = domain_2, domain_1

        key = domain_1 + '_+_' + domain_2
        matrix = coOccDict[key]
        domain_1_keys = list(nodeObj_Dict[domain_1].keys())
        domain_2_keys = list(nodeObj_Dict[domain_2].keys())
        args = []
        for i_idx, j_idx in product(domain_1_keys, domain_2_keys):
            count = matrix[i_idx][j_idx]
            if count == 0: continue
            args.append([nodeObj_Dict[domain_1][i_idx], nodeObj_Dict[domain_2][j_idx], count])

        G.add_weighted_edges_from(args)
        print('Number of nodes and edges :: ', G.number_of_nodes(), G.number_of_edges())
        save_graph(G, graph_file)
        return G


def get_graph_W_transaction_nodes(G, df):
    global id_col
    global nodeObj_Dict
    graph_file = 'nx_G1.pkl'
    G1 = read_graph(graph_file)

    if G1 is not None:
        return G1

    list_record_ids = list(df[id_col])
    create_transaction_nodes(list_record_ids)
    _list_obj = list(nodeObj_Dict[id_col].values())
    transactionID_2_serialID_dict = {e[0]: e[1] for e in enumerate(list(nodeObj_Dict[id_col].keys()), 0)}
    G.add_nodes_from(_list_obj)
    list_edge_types = read_schema()
    list_edge_types = [_ for _ in list_edge_types if id_col in _]

    num_jobs = multiprocessing.cpu_count()
    for edge in list_edge_types:
        print(edge)
        domain_1 = edge[0]
        domain_2 = edge[1]
        tmp_df = df[[domain_1, domain_2]]

        def aux_func_01(row, domain_1, domain_2):
            global nodeObj_Dict
            print(row)
            i = nodeObj_Dict[domain_1][row[domain_1]]
            j = nodeObj_Dict[domain_2][row[domain_2]]
            G.add_egdes_from([i,j])
        from pandarallel import pandarallel
        pandarallel.initialize()

        tmp_df.parallel_apply(
            aux_func_01,
            axis =1,
            args=( domain_1, domain_2, )
        )
        # edge_list = Parallel(num_jobs)(
        #     delayed(aux_func_01)
        #     (row, domain_1, domain_2, ) for _, row in tmp_df.iterrows())
        # print(edge_list)
        # G.add_egdes_from(edge_list)
    save_graph(G, graph_file)
    return G
