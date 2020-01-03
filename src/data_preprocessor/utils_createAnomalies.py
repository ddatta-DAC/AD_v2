import pandas as pd
import numpy as np
import hashlib
import os
import sys
import glob
import pickle

'''
Append a hash to speed up processing
'''


def get_hash_aux(row, id_col):
    row_dict = row.to_dict()
    del row_dict[id_col]
    _str = '_'.join([str(_) for _ in row_dict.values()])
    _str = str.encode(_str)
    str_hash = hashlib.md5(_str).hexdigest()
    return str_hash


'''
Add a 'hash' column, hash type : md5 , str 
'''


def add_hash(df, id_col):
    df['hash'] = df.apply(
        get_hash_aux,
        axis=1,
        args=(id_col,)
    )
    return df


'''
Check if a hash value is present in the 'hash' column of the given dataframe
'''


def is_duplicate(ref_df, hash_val):
    if len(ref_df.loc[ref_df['hash'] == hash_val]) > 0: return True
    return False


'''
Modify the id_col
'''


def aux_modify_id(value_id_col, suffix):
    return int(str(value_id_col) + str(suffix))


'''
Divide the given dataframe into chunks for concurrency
'''


def chunk_df(df, num_chunks):
    chunk_len = int(len(df) // num_chunks)
    list_df_chunks = np.split(
        df.head(chunk_len * (num_chunks - 1)), num_chunks - 1
    )
    end_len = len(df) - chunk_len * (num_chunks - 1)
    list_df_chunks.append(df.tail(end_len))
    return list_df_chunks


def create_coocc_matrix(df, col_1, col_2):
    set_elements_1 = set(list(df[col_1]))
    set_elements_2 = set(list(df[col_2]))
    count_1 = len(set_elements_1)
    count_2 = len(set_elements_2)
    coocc = np.zeros([count_1, count_2])
    df = df[[col_1, col_2]]
    new_df = df.groupby([col_1, col_2]).size().reset_index(name='count')

    for _, row in new_df.iterrows():
        i = row[col_1]
        j = row[col_2]
        coocc[i][j] = row['count']

    print('Col 1 & 2', col_1, col_2, coocc.shape, '>>', (count_1, count_2))
    return coocc


'''
Create co-occurrence between entities using training data. 
Returns a dict { Domain1_+_Domain2 : __matrix__ }
Domain1 and Domain2 are sorted lexicographically
'''


def get_coOccMatrix_dict(df, id_col):
    columns = list(df.columns)
    columns.remove(id_col)
    columns = list(sorted(columns))
    columnWise_coOccMatrix_dict = {}

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_1 = columns[i]
            col_2 = columns[j]
            key = col_1 + '_+_' + col_2
            res = create_coocc_matrix(df, col_1, col_2)
            columnWise_coOccMatrix_dict[key] = res
    return columnWise_coOccMatrix_dict


'''
Given a dictionary of { domain:entity , ... } , find the number of rows containing the same in the reference df
'''


def find_pattern_count(domainEntity_dict, ref_df):
    global id_col
    query_str = []

    for _c, _i in domainEntity_dict.items():
        query_str.append(' ' + _c + ' == ' + str(_i))
    query_str = ' & '.join(query_str)
    res_query = ref_df.query(query_str)
    return len(res_query)


'''
Remove duplicates from a list of dictionaries
'''


def dedup_list_dictionaries(list_domEntDictionaries):
    result = []
    set_hash_values = set()
    for _dict in list_domEntDictionaries:
        keys = sorted(list(_dict.keys()))
        hash_input = ''
        for k in keys:
            hash_input = hash_input + str(k) + '_' + str(_dict[k])
        hash_val = str.encode(hashlib.md5(hash_input).hexdigest())
        if hash_val not in set_hash_values:
            result.append(_dict)
            set_hash_values = set_hash_values.union(hash_val)

    return result
