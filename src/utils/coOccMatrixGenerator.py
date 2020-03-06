import pandas as pd
import numpy as np


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