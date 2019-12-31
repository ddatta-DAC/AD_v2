import pandas as pd
import os
import sys
import numpy as np
import pickle
from scipy.sparse import csr_matrix

def create_coocc_matrix(df, col_1,col_2):
    set_elements_1 = set(list(df[col_1]))
    set_elements_2 = set(list(df[col_2]))


