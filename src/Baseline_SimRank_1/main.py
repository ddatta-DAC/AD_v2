import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')



# ---------------
# Algorithm ::
# Create a network
# Calculate SimRank between the Transaction nodes
#
# With partially labelled data - Train a classifier
# Classify points on the unlabelled data (transaction instances : where features are entities + anomaly scores )
# Set final label as
# Sign ( lambda * Weighted(similarity based) of labels of its K nearest (labelled) neighbors + (1-lambda) predicted label )
# ----------------


