#!/usr/bin/env python
#-*- coding: utf-8 -*-

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
try:
    from src.data_fetcher import data_fetcher_v2 as data_fetcher
except:
    from data_fetcher import data_fetcher_v2 as data_fetcher

model_use_data_DIR = './model_use_data'
nodeObj_dict_file = 'nodeObj_dict.pkl'
REFRESH_NODES  = False
id_col = 'PanjivaRecordID'



def get_domain_dims(DIR):

    with open(
            os.path.join(
                './../../generated_data_v1/',
                DIR,
                'domain_dims.pkl'
            ), 'rb') as fh:
        domain_dims = pickle.load(fh)
    return domain_dims

# Create graph
class node:
    id = 0
    @staticmethod
    def get_id():
        _id = node.id
        node.id = node.id + 1
        return _id

    def __init__(self, node_type , e_id):
        self.node_type =  node_type
        self.id = node.get_id()
        self.e_id = e_id



# Create the nodes first


# Add edges
njobs = multiprocessing.cpu_count()

def create_transaction_nodes(list_record_ids, nodeObj_dict):
    global id_col
    domain = id_col
    nodeObj_dict[domain] = defaultdict()
    for e_id in sorted(list_record_ids):
        node_obj = node(node_type=domain, e_id=e_id)
        nodeObj_dict[domain][e_id] = node_obj
    return nodeObj_dict

def create_entity_nodes(domain_dims, nodeObj_dict):
    for domain,size in domain_dims.items():
        nodeObj_dict[domain] = defaultdict()
        for e_id in range(size):
            node_obj = node( node_type=domain, e_id = e_id)
            nodeObj_dict[domain][e_id] = node_obj
    return nodeObj_dict

def get_all_nodes ( data_df, domain_dims, DIR):
    global id_col
    global REFRESH_NODES
    global nodeObj_dict_file
    global model_use_data_DIR

    if not os.path.exists(model_use_data_DIR):
        os.mkdir(model_use_data_DIR)
    if not  os.path.exists(os.path.join(model_use_data_DIR,DIR)):
        os.mkdir(os.path.join(model_use_data_DIR,DIR))

    f_path = os.path.join(model_use_data_DIR,DIR, nodeObj_dict_file)

    if os.path.exists(f_path):
        with open(f_path,'rb') as fh:
            nodeObj_dict = pickle.load(fh)
            return nodeObj_dict


    nodeObj_dict = defaultdict()
    list_record_ids = list(data_df[id_col])
    nodeObj_dict = create_entity_nodes(domain_dims, nodeObj_dict)
    nodeObj_dict = create_transaction_nodes(list_record_ids, nodeObj_dict)
    with open(f_path, 'wb') as fh:
        pickle.dump(nodeObj_dict, fh, pickle.HIGHEST_PROTOCOL)
    return nodeObj_dict

def get_training_data(DIR):
    SOURCE_DATA_DIR = './../../generated_data_v1'
    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR, DIR)
    return data


DIR = 'us_import1'
domain_dims = get_domain_dims(DIR)
df = get_training_data(DIR)
nodeObj_dict = get_all_nodes ( df, domain_dims, DIR)
