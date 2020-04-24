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
from multiprocessing import Pool
from itertools import product

try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher

nodeObj_dict_file = 'nodeObj_dict_00.pkl'
REFRESH_NODES = False
id_col = 'PanjivaRecordID'
nodeObj_Dict = None
model_use_data_DIR = None
mapping_df_file = None
serial_mapping_df = None
serialized_data_df_file = None

# ------------------------------------------------------------3
# Call this to set up global variables
# ------------------------------------------------------------
def initialize(_dir, _model_use_data_DIR):
    global DIR
    global model_use_data_DIR
    global mapping_df_file
    global serialized_data_df_file
    global serialized_data_df_T_file

    mapping_df_file = 'Serialized_Mapping.csv'


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
    mapping_df_file = os.path.join(model_use_data_DIR, mapping_df_file)

    serialized_data_df_file = os.path.join(model_use_data_DIR, 'serialized_data.csv')
    serialized_data_df_T_file = os.path.join(model_use_data_DIR, 'serialized_data_T.csv')
    return


# ------------------------------------------------------------
def get_training_data(DIR):
    SOURCE_DATA_DIR = './../../generated_data_v1'
    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR, DIR)
    return data


def get_domain_dims():
    global DIR
    with open(
            os.path.join(
                './../../generated_data_v1/',
                DIR,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)
    return domain_dims

def convert2_Serial_ID(_row, cols):
    global serial_mapping_df

    row = _row.copy()
    for c in cols:
        val = row[c]
        res = list(
            serial_mapping_df.loc[
                (serial_mapping_df['Domain'] == c) &
                (serial_mapping_df['Entity_ID'] == val)]
            ['Serial_ID']
        )
        if len(res) > 0 :
            row[c] = res[0]
    return row

def serialize_data( df ):
    global DIR
    global mapping_df_file
    global id_col
    global serial_mapping_df
    global serialized_data_df_file

    domain_dims = get_domain_dims()
    print( 'Mapping to serial ids file ::', mapping_df_file)

    if not os.path.exists(mapping_df_file):
        prev_count = 0
        res = []
        for dn, ds in domain_dims.items():
            for eid in range(ds):
                r = [dn, eid, eid + prev_count]
                res.append(r)
            prev_count += ds

        serial_mapping_df = pd.DataFrame(
            data=res,
            columns=['Domain', 'Entity_ID', 'Serial_ID']
        )

        print(mapping_df_file)
        serial_mapping_df.to_csv(
            mapping_df_file,
            index=False
        )
    else:
        serial_mapping_df = pd.read_csv(mapping_df_file, index_col=None)

    cols = list(df.columns)
    cols.remove(id_col)
    print (cols)
    if not os.path.exists(serialized_data_df_file):
        serialized_df = df.parallel_apply(
            convert2_Serial_ID,
            axis=1,
            args=(cols,)
        )
        serialized_df.to_csv(serialized_data_df_file,index=False)
    else:
        serialized_df = pd.read_csv(serialized_data_df_file,index_col=None)
    return  serialized_df


def create_entity_nodes(domain_dims, nodeObj_dict):
    global serial_mapping_df
    for domain, size in domain_dims.items():
        Serial_ID_list = list(serial_mapping_df.loc[serial_mapping_df['Domain']==domain])
        nodeObj_dict[domain] = Serial_ID_list
    return nodeObj_dict



# def create_transaction_nodes(list_record_ids):
#     global id_col
#     global nodeObj_Dict
#     domain = id_col
#     nodeObj_Dict[domain] = defaultdict()
#     for e_id in sorted(list_record_ids):
#         node_obj = node(node_type=domain, e_id=e_id)
#         nodeObj_Dict[domain][e_id] = node_obj
#     return




# ---------------------------
# Use the training data to create nodes
# Initially create entity nodes only
# ---------------------------
def get_base_nodeObj_dict(
        domain_dims
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


def save_graph(G, graph_file):
    global model_use_data_DIR
    graph_obj_path = os.path.join(model_use_data_DIR, graph_file)
    nx.write_gpickle(G, graph_obj_path, protocol=4)

    return


def read_graph(graph_file):
    global model_use_data_DIR
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
    global DIR
    global serial_mapping_df

    serialized_df = serialize_data(df)
    graph_file = 'nx_G00.pkl'

    G0 = read_graph(graph_file)
    if G0 is not None:
        return G0

    nodeObj_Dict = get_base_nodeObj_dict(domain_dims)
    print(domain_dims)

    list_edges = read_schema()

    G = nx.Graph()
    # Add in the nodes
    # for domain, list_nodes in nodeObj_Dict.items():
    #     G.add_nodes_from(list_nodes)

    # Add edges
    n_jobs = multiprocessing.cpu_count()


    for edge in list_edges:
        if id_col in edge: continue
        print(edge)
        domain_1 = edge[0]
        domain_2 = edge[1]
        tmp_df = serialized_df[[domain_1,domain_2]]
        tmp_df = pd.DataFrame(
            tmp_df.groupby([domain_1,domain_2]).size().reset_index(name='weight')
        )

        def aux_f_01(row,d1,d2):
            return (row[d1],row[d2],row['weight'])

        res =  tmp_df.parallel_apply(
            aux_f_01,axis=1,args=(domain_1,domain_2,)
        )
        print(len(res))
        G.add_weighted_edges_from(res)
        print('Number of nodes and edges :: ', G.number_of_nodes(), G.number_of_edges())
        save_graph(G, graph_file)
        return G


def convert_to_serial_id (df):
    global serial_mapping_df

    return


# -----------------------
# Inputs:
# 1. Initial graph with entity nodes only
# 2. Test df
# -----------------------
def get_graph_W_transaction_nodes(G, df):
    global id_col
    global nodeObj_Dict
    global serial_mapping_df
    global serialized_data_df_T_file

    graph_file = 'nx_G01.pkl'
    G1 = read_graph(graph_file)
    if G1 is not None:
        return G1

    list_record_ids = list(sorted(df[id_col]))
    # Convert the transaction ids and add them to serial_mapping_df
    cur_serial_id = max(list(serial_mapping_df['Serial_ID']))+1
    list_node_serial_ids = []
    _data = []

    for _id in list_record_ids:
        _data.append([ id_col, _id, cur_serial_id ])
        list_node_serial_ids.append(cur_serial_id)
        cur_serial_id += 1

    new_df = pd.DataFrame(data =_data, columns=['Domain', 'Entity_ID', 'Serial_ID'])
    serial_mapping_df = serial_mapping_df.append(
        new_df,
        ignore_index=True
    )

    G.add_nodes_from(list_node_serial_ids)
    # Now Serialized df is the test data
    if not os.path.exists(serialized_data_df_T_file):
        cols = list(df.columns)
        serialized_df = df.parallel_apply(
            convert2_Serial_ID,
            axis=1,
            args=(cols,)
        )
        serialized_df.to_csv(serialized_data_df_T_file, index=False)
    else:
        serialized_df = pd.read_csv(
            serialized_data_df_T_file,
            index_col=None
        )

    list_edge_types = read_schema()
    list_edge_types = [_ for _ in list_edge_types if id_col in _]

    num_jobs = multiprocessing.cpu_count()
    for edge in list_edge_types:
        print(edge)
        domain_1 = edge[0]
        domain_2 = edge[1]
        tmp_df = serialized_df[[domain_1, domain_2]]
        tmp_df = pd.DataFrame(
            tmp_df.groupby([domain_1, domain_2]).size().reset_index(name='weight')
        )

        def aux_f_02(row, d1, d2):
            return (row[d1], row[d2],1)

        res = tmp_df.parallel_apply(
            aux_f_02, axis=1, args=(domain_1, domain_2,)
        )
        print(len(res))
        G.add_weighted_edges_from(res)
        print('Number of nodes and edges :: ', G.number_of_nodes(), G.number_of_edges())

    save_graph(G, graph_file)
    return G

