import pickle
import pandas as pd


def get_domain_dims(dd_file_path):
    with open(dd_file_path, 'rb') as fh:
        domain_dims = pickle.load(fh)
    _tmpDF = pd.DataFrame.from_dict(domain_dims,orient='index')
    _tmpDF = _tmpDF.reset_index()
    _tmpDF = _tmpDF.rename(columns={'index':'domain'})
    _tmpDF = _tmpDF.sort_values(by=['domain'])
    res = { k:v for k,v in zip(_tmpDF['domain'], _tmpDF[0])}
    return res