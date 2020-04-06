import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')
try:
    from utils import plotter
except:
    from src.utils import plotter

try:
    from .metapath2vec_1 import model_mp2v_1
except:
    from metapath2vec_1 import model_mp2v_1

try:
    from . import network_data_loader
except:
   import network_data_loader



# -------------------------------------------- #

domain_dims = network_data_loader.get_domain_dims()
num_entities = sum(list(domain_dims.values()))
obj = model_mp2v_1.model()

obj.build(
    emb_dim=128,
    num_entities=num_entities,
    num_neg_samples=10,
    context_size=2,
    batch_size=256,
    num_epochs=10
)

x_t, x_c, x_ns = data_loader.fetch_model_data_m2pv_1()
y = obj.train_model(x_t, x_c, x_ns)
# x = range(len(y))
# plotter.get_general_plot(
#     x,y
# )
