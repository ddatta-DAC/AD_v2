import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./../..')
sys.path.append('./..')
from pandarallel import pandarallel
pandarallel.initialize()


# ------------------------------------ #

# Input :
# Stage 1 :
# Training data to create network
# Metapaths
# Stage 2
# DataFrame of [ Test Transactions ids, Scores , Entities ]
# ------------------------------------- #
