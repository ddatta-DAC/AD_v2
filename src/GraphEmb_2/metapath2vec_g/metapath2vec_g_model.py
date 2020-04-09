import gensim
import pandas as pd
import numpy as np
import os
import argparse
import pickle
import sys
from gensim.similarities.index import AnnoyIndexer
sys.path.append('./../..')
sys.path.append('./..')
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from pandarallel import pandarallel
pandarallel.initialize()
from src.utils import coOccMatrixGenerator



# ----------------------------- #
domain_dims = None
SOURCE_DATA_DIR_1 = './../../../generated_data_v1'
MODEL_INP_DATA_DIR = './data'
id_col = 'PanjivaRecordID'
coOccDict = None
serial_mapping_df = None
metapath2vec_data_DIR = None

def get_data():
    global SOURCE_DATA_DIR_1
    global DIR
    global MODEL_INP_DATA_DIR
    global id_col
    global coOccDict
    global serial_mapping_df
    global domain_dims
    global metapath2vec_data_DIR
    data = data_fetcher.get_train_x_csv(SOURCE_DATA_DIR_1, DIR)
    domain_dims = data_fetcher.get_domain_dims(SOURCE_DATA_DIR_1, DIR)
    print(domain_dims)

    coOccDict_file = os.path.join(MODEL_INP_DATA_DIR, 'co_occ_dict.pkl')
    if not os.path.exists(coOccDict_file):
        coOccDict = coOccMatrixGenerator.get_coOccMatrix_dict(
            data, id_col
        )
        with open(coOccDict_file, 'wb') as fh:
            pickle.dump(coOccDict, fh, pickle.HIGHEST_PROTOCOL)
    else:
        with open(coOccDict_file, 'rb') as fh:
            coOccDict = pickle.load(fh)

    mapping_df_file = 'Serialized_Mapping.csv'
    if not os.path.exists(MODEL_INP_DATA_DIR):
        os.mkdir(MODEL_INP_DATA_DIR)

    mapping_df_file = os.path.join(MODEL_INP_DATA_DIR, mapping_df_file)

    if not os.path.exists(mapping_df_file):
        prev_count = 0
        res = []
        for dn, ds in domain_dims.items():
            for eid in range(ds):
                r = [dn, eid, eid + prev_count]
                res.append(r)
            prev_count += ds

        serial_mapping_df = pd.DataFrame(
            data=res,
            columns=['Domain', 'Entity_ID', 'Serial_ID']
        )

        print(mapping_df_file)
        serial_mapping_df.to_csv(
            mapping_df_file,
            index=False
        )
    else:
        serial_mapping_df = pd.read_csv(mapping_df_file, index_col=None)

    def convert(_row, cols):
        row = _row.copy()
        for c in cols:
            val = row[c]
            res = list(
                serial_mapping_df.loc[
                    (serial_mapping_df['Domain'] == c) &
                    (serial_mapping_df['Entity_ID'] == val)]
                ['Serial_ID']
            )
            row[c] = res[0]
        return row

    serialized_data_file = os.path.join(MODEL_INP_DATA_DIR, 'serialized_data.csv')

    if not os.path.exists(serialized_data_file):
        cols = list(data.columns)
        cols.remove(id_col)
        serialized_data = data.parallel_apply(
            convert,
            axis=1,
            args=(cols,)
        )
        del serialized_data[id_col]
        serialized_data.to_csv(serialized_data_file, index=False)
    else:
        serialized_data = pd.read_csv(serialized_data_file, index_col=None)

    data_text_file = parse_into_sentence(serialized_data)
    return data_text_file


def parse_into_sentence(data_df):
    global MODEL_INP_DATA_DIR

    def sentencify(row):
        vals = row.values
        np.random.shuffle(vals)
        res = ' '.join([str(_) for _ in vals])
        return res

    data_text_file = os.path.join(
        MODEL_INP_DATA_DIR,
        'text_data.txt'
    )
    if os.path.exists(data_text_file):
        return data_text_file

    sentencified = data_df.parallel_apply(
        sentencify,
        axis=1
    )
    result = sentencified.values
    # write to a text file

    with open(data_text_file, 'w') as fh:
        fh.writelines('\n'.join(result))
    return data_text_file


# ------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--DIR', choices=['us_import1', 'us_import2', 'us_import3'],
    default='us_import1'
)

args = parser.parse_args()
DIR = args.DIR
# -------------------------- #
REFRESH = False
data_text_file = get_data()

model_file = 'gensim_model.dat'
if REFRESH or not os.path.exists(model_file):
    sentences = LineSentence(data_text_file)
    model = Word2Vec(
        sentences=sentences,
        size=128,
        window=5,
        min_count=1,
        sample=0.01,
        seed=100,
        workers=8,
        min_alpha=0.0001,
        negative=25,
        ns_exponent=0.75,
        iter=100,
        null_word=-1
    )

    model.save(model_file)
else:
    model = Word2Vec.load(model_file)


def check_NN(model):
    global domain_dims
    global coOccDict
    global serial_mapping_df
    annoy_index = AnnoyIndexer(model, 128)

    def get_Serial_ID(domain, e_id):
        res = list(serial_mapping_df.loc[
                       (serial_mapping_df['Domain'] == domain) &
                       (serial_mapping_df['Entity_ID'] == e_id)
                       ]['Serial_ID'])[0]
        return res

    def get_Entity_ID(serial_ID):
        d = list(serial_mapping_df.loc[
                     (serial_mapping_df['Serial_ID'] == serial_ID)
                 ]['Domain'])[0]
        e = list(serial_mapping_df.loc[
                     (serial_mapping_df['Serial_ID'] == serial_ID)
                 ]['Entity_ID'])[0]
        return e, d

    # select 3 entites from each domain
    for dom in domain_dims.keys():
        sampled_entities = np.random.choice(range(domain_dims[dom]), size=3, replace=False)

        for e1 in sampled_entities:
            s_id = get_Serial_ID(dom, e1)
            print('>>>', dom, e1)
            vector = model.wv[str(s_id)]
            print("Approximate Neighbors")
            approximate_neighbors = model.wv.most_similar(
                [vector],
                topn=11,
                indexer=annoy_index
            )

            for n_s_id in approximate_neighbors[1:]:

                n_s_id = int(n_s_id[0])
                e, d = get_Entity_ID(n_s_id)
                if dom == d : continue
                if d < dom:
                    key = d + '_+_' + dom
                    align = 'r'
                else:
                    key = dom + '_+_' + d
                    align = 'c'

                matrix = coOccDict[key]
                if align == 'r':
                    arr = matrix[e, :]
                else:
                    arr = matrix[:, e]
                print('Domain ', d, 'Entity ', e,' Count', arr[e1])


check_NN(model)
