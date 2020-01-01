import pandas as pd
import os
import sys
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from itertools import combinations
from joblib import Parallel, delayed


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

    print('Col 1 & 2', col_1, col_2, coocc.shape, '>>', (count_1,count_2) )
    return coocc


'''
Remove the rows in test set where any apir of entities do not co-occur in the training set.
'''

# Returns False if there is supurious pairwise-co-occurrence
def aux2_check(row, columnWise_coOcc_array_dict, pair_list):
    row_dict = row.to_dict()
    for _pair in pair_list:
        _key = '_+_'.join(_pair)
        i = row_dict[_pair[0]]
        j = row_dict[_pair[1]]

        if columnWise_coOcc_array_dict[_key][i][j] == 0:
            return False
    return True


def aux_check(df, columnWise_coOcc_array_dict, id_col):
    print('In Auxillary check ...')
    columns = list(df.columns)
    columns.remove(id_col)
    pair_list = [list(sorted(_pair)) for _pair in combinations(columns, 2)]
    df['valid'] = None
    df['valid'] = df.apply(
        aux2_check,
        axis=1,
        args=(columnWise_coOcc_array_dict, pair_list,)
    )
    res_df = pd.DataFrame(df.loc[df['valid'] == True], copy=True)
    del res_df['valid']
    return res_df


def remove_order1_spurious_coocc(
        train_df,
        test_df,
        id_col='PanjivaRecordID'
):
    print('In remove_order1_spurious_coocc ::')
    columns = list(train_df.columns)
    columns.remove(id_col)
    columns = list(sorted(columns))
    columnWise_coOcc_array_dict = {}

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_1 = columns[i]
            col_2 = columns[j]
            key = col_1 + '_+_' + col_2
            res = create_coocc_matrix(train_df, col_1, col_2)
            columnWise_coOcc_array_dict[key] = res

    num_chunks = 40
    chunk_len = int(len(test_df) // num_chunks)

    list_df_chunks = np.split(
        test_df.head(chunk_len * (num_chunks - 1)),
        num_chunks - 1
    )

    end_len = len(test_df) - chunk_len * (num_chunks - 1)
    list_df_chunks.append(test_df.tail(end_len))
    print(' Chunk lengths ->', [len(_) for _ in list_df_chunks])

    list_dedup_df = Parallel(n_jobs=num_chunks)(
        delayed(aux_check)(
            target_df, columnWise_coOcc_array_dict, id_col
        ) for target_df in list_df_chunks
    )
    print('Post cleaning chunk lengths ->', [len(_) for _ in list_dedup_df])

    new_test_df = None
    for _df in list_dedup_df:
        if new_test_df is None:
            new_test_df = _df
        else:
            new_test_df = new_test_df.append(_df, ignore_index=True)

    print(' After deduplication :: ', len(new_test_df))
    return new_test_df
