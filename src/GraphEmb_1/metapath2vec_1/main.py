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

# -------------------------------------------- #

num_entities = 100
obj = model()
obj.build(
    emb_dim=250,
    num_entities=num_entities,
    num_neg_samples=5,
    context_size=3,
    batch_size=6
)


a = np.random.randint(num_entities,size=[1000])
b = np.random.randint(num_entities,size=[1000,3])
c = np.random.randint(num_entities,size=[1000,5,3])

y = obj.train_model(a,b,c)
x = range(len(y))
f = plotter.get_general_plot(
    x,y
)
print(f)