import tensorflow as tf
import keras.backend as K
import numpy as np
import os
from keras import Model
from keras.layers import Input, Embedding, Dot, Reshape, Add
from keras.layers import Lambda

X_ij_max = None
# =================================
# Co-occurrence based embedding model
# =================================
def get_model(
    domain_dimesnsions = None,
    num_domains = 4,
    embed_dim = 16,
    _X_ij_max = None
):

    global X_ij_max
    X_ij_max = _X_ij_max
    embedding_layer = []
    bias_layer = []

    input_layer = Input(
        shape=(num_domains,)
    )

    # =======================
    # Input record
    # =======================
    split_input_record = Lambda(
        lambda x:
        tf.split(
            x,
            num_or_size_splits=num_domains,
            axis=-1)
        ,
        name='split_layer'
    )(input_layer)

    for i in range(num_domains):
        emb_i = Embedding(
            input_dim = domain_dimesnsions[i],
            output_dim= embed_dim,
            embeddings_initializer='random_uniform',
            name='embedding_w_'+str(i)
        )(split_input_record[i])
        embedding_layer.append(emb_i)

        bias_i = Embedding(
            input_dim = domain_dimesnsions[i],
            output_dim=1,
            input_length=1,
            embeddings_initializer='random_uniform',
            name= 'embedding_b_'+str(i)
        )(split_input_record[i])
        bias_layer.append(bias_i)

    y_pred = []

    for i in range(num_domains):
        for j in range(i+1,num_domains):
            w_i__w_j = Dot(axes=-1)([
                embedding_layer[i],
                embedding_layer[j]
            ])
            w_i__w_j = Reshape(target_shape=(1,))(w_i__w_j)
            pred_logXij = Add()([w_i__w_j, bias_layer[i],bias_layer[j]])
            pred_logXij = Reshape(target_shape=(1,))(pred_logXij)
            y_pred.append(pred_logXij)

    y_pred_stacked = Lambda(
        lambda x:
        tf.stack(
            x,
            axis=1
        ),
        name='stack_layer'
    )(y_pred)

    y_pred_final = Lambda(
        lambda x:
        tf.squeeze(
            x,
            axis=-1
        ),
        name='squeeze_layer'
    )(y_pred_stacked)

    model = Model(
        input_layer,
        y_pred_final
    )
    model.compile(
        loss = custom_loss_function,
        optimizer='adam'
    )

    return model

def custom_loss_function(
        y_true,
        y_pred
):
    global X_ij_max
    a = 0.75
    epsilon = 0.000001

    _err1 = K.square(y_pred - tf.math.log(y_true + epsilon))
    _scale1 = K.pow(
        K.clip(y_true / X_ij_max, 0.0, 1.0),
        a
    )
    loss = _scale1 * _err1
    return K.sum(
        loss,
        axis=-1
    )

def train_model(
        model,
        x,
        y_true,
        file_save_loc,
        epochs=100
):
    model.summary()
    model.fit(
        x=x,
        y=y_true,
        batch_size=64,
        epochs=epochs,
        verbose=1,
        shuffle=True
    )
    save_model(model,file_save_loc)

    return model

def save_model(model, file_save_loc):

    for layer in model.layers:
        print(layer.name)
        if 'embedding_w' in layer.name:
            f_path = os.path.join( file_save_loc, layer.name + ".npy")
            np.save(f_path, arr=layer.get_weights()[0])

