import gensim
import numpy as np
import os
import sys
from gensim.similarities.index import AnnoyIndexer
sys.path.append('./../..')
sys.path.append('./..')
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from pandarallel import pandarallel
pandarallel.initialize()
import multiprocessing

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

def get_model_obj(
    corpus_txt_file_path,
    emb_size = 64,
    window_size = 8,
    model_save_path = 'mp2v_gensim.dat',
    num_neg_samples = 10,
    load_saved = True
):
    num_jobs = multiprocessing.cpu_count()
    if os.path.exists(model_save_path) and load_saved:
        model = Word2Vec.load(model_save_path)
    else:
        model = Word2Vec(
            sentences = LineSentence(corpus_txt_file_path),
            size=emb_size,
            window=window_size,
            min_count=1,
            sample=0.01,
            seed=100,
            workers=num_jobs,
            min_alpha=0.0001,
            negative=num_neg_samples,
            ns_exponent=0.75,
            iter=100,
            null_word=-1
        )
        model.save(model_save_path)

    return model


#
# def check_NN(model):
#     global domain_dims
#     global coOccDict
#     global serial_mapping_df
#     annoy_index = AnnoyIndexer(model, 128)
#
#     def get_Serial_ID(domain, e_id):
#         res = list(serial_mapping_df.loc[
#                        (serial_mapping_df['Domain'] == domain) &
#                        (serial_mapping_df['Entity_ID'] == e_id)
#                        ]['Serial_ID'])[0]
#         return res
#
#     def get_Entity_ID(serial_ID):
#         d = list(serial_mapping_df.loc[
#                      (serial_mapping_df['Serial_ID'] == serial_ID)
#                  ]['Domain'])[0]
#         e = list(serial_mapping_df.loc[
#                      (serial_mapping_df['Serial_ID'] == serial_ID)
#                  ]['Entity_ID'])[0]
#         return e, d
#
#     # select 3 entites from each domain
#     for dom in domain_dims.keys():
#         sampled_entities = np.random.choice(range(domain_dims[dom]), size=3, replace=False)
#
#         for e1 in sampled_entities:
#             s_id = get_Serial_ID(dom, e1)
#             print('>>>', dom, e1)
#             vector = model.wv[str(s_id)]
#             print("Approximate Neighbors")
#             approximate_neighbors = model.wv.most_similar(
#                 [vector],
#                 topn=11,
#                 indexer=annoy_index
#             )
#
#             for n_s_id in approximate_neighbors[1:]:
#
#                 n_s_id = int(n_s_id[0])
#                 e, d = get_Entity_ID(n_s_id)
#                 if dom == d : continue
#                 if d < dom:
#                     key = d + '_+_' + dom
#                     align = 'r'
#                 else:
#                     key = dom + '_+_' + d
#                     align = 'c'
#
#                 matrix = coOccDict[key]
#                 if align == 'r':
#                     arr = matrix[e, :]
#                 else:
#                     arr = matrix[:, e]
#                 print('Domain ', d, 'Entity ', e,' Count', arr[e1])
#
#
# check_NN(model)
