import pandas as pd
import os
import sys
import numpy as np
import pickle
from scipy.sparse import csr_matrix

def create_coocc_matrix(df, col_1,col_2):
    set_elements_1 = set(list(df[col_1]))
    set_elements_2 = set(list(df[col_2]))
    count_1 = len(set_elements_1)
    count_2 = len(set_elements_2)
    coocc = np.zeros([count_1,count_2])
    coocc = csr_matrix(coocc)

    df = df[[col_1,col_2]]



