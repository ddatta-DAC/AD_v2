'''
Assumption that test set is normal, may not be true.
This post processor is run on the test data to ensure test_data has no pairwise spurious co-occurrence.
Remove the rows in test set where any pair of entities do not co-occur in the training set.
AKA remove primary spurious co-occurrence
'''

import pandas as pd
import os
import sys
import numpy as np
import pickle
from itertools import combinations
from joblib import Parallel, delayed

try:
    from . import utils_createAnomalies as utils_local

except:
    import utils_createAnomalies as utils_local

'''

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


'''
This function removes the row with spurious pairwise co-occurrences
'''
'''
Assumption that test set is normal, may not be true.
This post processor is run on the test data to ensure test_data has no pairwise spurious co-occurrence.
Remove the rows in test set where any pair of entities do not co-occur in the training set.
AKA remove primary spurious co-occurrence
'''

import pandas as pd
import os
import sys
import numpy as np
import pickle
from itertools import combinations
from joblib import Parallel, delayed

try:
    from . import utils_createAnomalies as utils_local

except:
    import utils_createAnomalies as utils_local

'''

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


'''
This function removes the row with spurious pairwise co-occurrences
'''


def remove_order1_spurious_coocc(
        train_df,
        test_df,
        id_col='PanjivaRecordID'
):
    print('In remove_order1_spurious_coocc ::')
    columnWise_coOcc_array_dict = utils_local.get_coOccMatrix_dict(
        train_df,
        id_col
    )
    num_chunks = 40
    list_df_chunks = utils_local.chunk_df(
        test_df,
        num_chunks
    )

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


'''
Main function
Inputs: 
Train_df
Test_df
'''


def remove_order1_spurious_coocc(
        train_df,
        test_df,
        id_col='PanjivaRecordID',
        num_jobs=10
):
    print('In remove_order1_spurious_coocc ::')
    columnWise_coOcc_array_dict = utils_local.get_coOccMatrix_dict(
        train_df,
        id_col
    )
    num_chunks = num_jobs
    list_df_chunks = utils_local.chunk_df(
        test_df,
        num_chunks
    )

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
