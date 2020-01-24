import numpy as np
import os
import sys
import keras
import keras.backend as K
from keras import Model, Sequential, layers
from keras.layers import Lambda
import tensorflow as tf
from keras.layers import Input, Dense, Embedding, Dot, Reshape, Add, Average, Concatenate, TimeDistributed, \
    Bidirectional, LSTM, Multiply

# =============== Global variables ================== #
num_domains = None
domain_dims = None
input_emb_dim = None
lstm_dim = 32
ctx_dim_1 = 32
interaction_dim = 32
num_neg_samples = 10
save_dir = None

# ===================================================== #
def get_model(
        num_domains,
        domain_dims,
        domain_emb_wt,
        lstm_dim,
        interaction_layer_dim,
        context_dim,
        num_neg_samples=10,
        RUN_MODE='train',
        save_dir = None,
        model_signature = None
):

    # Dimension of the context vector
    ctx_dim_1 = context_dim
    # Dimension of the interaction layer
    interaction_dim = interaction_layer_dim
    # ---------------------------------
    # list of np arrays which are sorted by domain name lexicographically
    # ---------------------------------
    input_emb_dim = domain_emb_wt[0].shape[-1]
    n_timesteps = num_domains + 2

    def tf_stack(x):
        import tensorflow as tf

        return tf.stack(
            x,
            axis=1
        )

    def tf_squeeze(x):
        import tensorflow as tf

        x1 = tf.squeeze(
            x,
            axis=1
        )
        return x1

    def tf_split_dplus2(x):

        return tf.split(
            x,
            num_or_size_splits=n_timesteps,
            axis=1
        )

    def tf_reduce_sum(x):
        import tensorflow as tf
        return tf.math.reduce_sum(
            x,
            keepdims=False,
            axis=1
        )

    def split_squeeze_numDomains(x):
        import tensorflow as tf

        x1 = tf.split(
            x,
            num_or_size_splits=num_domains,
            axis=1
        )
        x2 = [tf.squeeze(_x2, axis=1) for _x2 in x1]
        return x2

    def tf_reduce_mean_kdims_axis1(x):
        import tensorflow as tf
        return tf.math.reduce_mean(
            x,
            axis=1,
            keepdims=True)

    def tf_split_squeeze_axis1(x):
        import tensorflow as tf
        x1 = tf.split(
            x,
            num_or_size_splits=x.shape[1],
            axis=1
        )
        x2 = [tf.squeeze(_x2, axis=1) for _x2 in x1]
        return x2

    def tf_reciprocal(x):
        import tensorflow as tf
        epsilon = .000001
        return tf.math.reciprocal(
            x + epsilon
        )

    def tf_sigmoid(x):
        import tensorflow as tf
        return tf.sigmoid(
            x
        )

    # ================= Define the weights ===================== #

    BD_LSTM_layer = Bidirectional(
        LSTM(
            units=lstm_dim,
            return_sequences=True
        ),
        input_shape=(n_timesteps, input_emb_dim),
        merge_mode=None
    )

    # Embedding layer for each domain
    list_Entity_Embed = [
        Embedding(
            input_dim=domain_dims[i],
            output_dim=input_emb_dim,
            embeddings_initializer=keras.initializers.Constant(value=domain_emb_wt[i]),
            name='entity_embedding_' + str(i)
        ) for i in range(num_domains)
    ]

    # -------------------------------------------
    # Dense layer for getting the Context vectors
    # keeping 1 for each domain
    # -------------------------------------------
    list_FNN_1 = [Dense(ctx_dim_1, activation='relu', use_bias=True) for _ in range(1, n_timesteps - 1)]
    list_FNN_2 = [Dense(interaction_dim, activation='relu') for _ in range(1, n_timesteps - 1)]
    # Dense layer for transforming the input vectors
    xform_Inp_FNN = [Dense(interaction_dim, activation=None, use_bias=True) for i in range(num_domains)]

    # ========================================================= #
    model = None
    def process(input_indices, _type='pos'):
        # Split the inputs
        split_input = Lambda(tf_split_squeeze_axis1)(input_indices)
        split_emb = []
        for i in range(num_domains):
            split_emb.append(list_Entity_Embed[i](split_input[i]))

        split_emb = [Lambda(tf_squeeze)(_) for _ in split_emb]
        # ------------
        # input embedding now has shape [ ?, num_domains, 256]
        # ------------
        input_emb = Lambda(tf_stack)(split_emb)

        mean_layer_op = Lambda(tf_reduce_mean_kdims_axis1)(input_emb)
        concat_layer = Concatenate(axis=1)(
            [mean_layer_op, input_emb, mean_layer_op]
        )
        n_timesteps = num_domains + 2
        bd_lstm = BD_LSTM_layer(concat_layer)

        # =========== #
        bd_lstm_fwd = bd_lstm[0]
        bd_lstm_bck = bd_lstm[1]
        # =========== #

        split_BL_F_op = Lambda(tf_split_dplus2)(bd_lstm_fwd)
        split_BL_B_op = Lambda(tf_split_dplus2)(bd_lstm_bck)
        split_BL_B_op = [Lambda(tf_squeeze)(_) for _ in split_BL_B_op]
        split_BL_F_op = [Lambda(tf_squeeze)(_) for _ in split_BL_F_op]
        print('After Bi-Directional LSTM Cur shape :', len(split_BL_B_op), split_BL_B_op[0].shape)

        # ----------- #
        # Context vector
        # ----------- #

        ctx_output = []
        for i in range(1, n_timesteps - 1):
            _left = split_BL_F_op[i - 1]
            _right = split_BL_B_op[i - 1]

            # Context vector
            ctx_concat = Concatenate(axis=-1)([_left, _right])
            ctx_mlp_layer1 = list_FNN_1[i - 1](ctx_concat)
            ctx_mlp_layer2 = list_FNN_2[i - 1](ctx_mlp_layer1)
            ctx_output.append(ctx_mlp_layer2)

        print(' Cur shape [Context vector]:', ctx_output[0].shape)
        # ============
        # Final output as sigmoid ( sum of Dot products between context vector and modified input)
        # ============

        # =--------
        # Calculate interaction of the context vector with input  
        # =-------- 
        input_layer_split = Lambda(split_squeeze_numDomains)(input_emb)
        interaction_layer_input = [
            xform_Inp_FNN[i](input_layer_split[i])
            for i in range(num_domains)
        ]
        # dot product
        dot_product = [Dot(axes=-1)(
            [interaction_layer_input[i], ctx_output[i]]
        ) for i in range(num_domains)]

        stacked_dot_op = Lambda(tf_stack)(dot_product)

        if _type == 'neg':
            stacked_dot_op = Lambda(tf_reciprocal)(stacked_dot_op)
            final_op = Lambda(tf_reduce_sum)(stacked_dot_op)
        else:
            final_op = Lambda(tf_reduce_sum)(stacked_dot_op)

        final_op = Lambda(tf_sigmoid)(final_op)
        return final_op

    # ---------- train and test graphs should be different, but same  weights ----------- #
    if RUN_MODE == 'train':

        # ========= TRAIN mode =========== #
        print('Run mode ', RUN_MODE)
        pos_input = Input(shape=(num_domains, 1))
        pos_op = process(pos_input, _type='pos')

        neg_input = Input(shape=(num_neg_samples, num_domains, 1), name='negative_samples')
        neg_input_list = Lambda(tf_split_squeeze_axis1)(neg_input)
        neg_ops = []

        for n_sample in neg_input_list:
            n_res = process(n_sample, _type='pos')
            neg_ops.append(n_res)

        final_pred = Lambda(tf_stack)(neg_ops)
        final_pred = Lambda(
            lambda x:
            tf.math.reduce_mean(x, axis=1, keepdims=False)
        )(final_pred)

        final_pred = Add()([final_pred, pos_op])
        inputs = [pos_input, neg_input]
        outputs = final_pred
        model = Model(
            inputs=inputs,
            outputs=outputs,
            name = model_signature
        )
        # ====== Fix embedding weights ======= #
        for l in model.layers:
            if 'entity_embedding_' in l.name:
                l.trainable = False

    elif RUN_MODE == 'test':
        # ========= TEST mode =========== #
        pos_input = Input(shape=(num_domains, 1))
        print('Run mode ', RUN_MODE)
        final_pred = process(pos_input, _type='pos')
        inputs = pos_input
        outputs = final_pred

        # ==================================== #
        model = Model(
            inputs=inputs,
            outputs=outputs
        )

        h5_file_name = model_signature + ".h5"
        model_weights_path = os.path.join(
            save_dir, h5_file_name
        )
        model.load_weights(model_weights_path)
        for l in model.layers:
            l.trainable = False

    # ============== Custom Loss function ==================== #
    # Maximize the objective , Minimize -(predicted_val)
    # ======================================================== #
    def custom_loss(y_true, y_pred):
        return -y_pred

    optimizer = keras.optimizers.Adagrad()
    model.compile(optimizer, loss=custom_loss)

    return model


# =========================
#  Save model
# =========================
def save_model(save_dir, model, model_signature):
    model_json = model.to_json()
    model_json_name = model_signature + '.json'
    f_path = os.path.join(save_dir, model_json_name)
    with open(f_path, "w") as json_file:
        json_file.write(model_json)

    # ----
    # serialize weights to HDF5
    # ----
    h5_file_name = model_signature + ".h5"
    f_path = os.path.join(save_dir, h5_file_name)
    model.save_weights(f_path)
    print(" >>>>  Saved model {} to disk".format(model_signature))
    return
# =====================================



# =====================================
#  Input to model:
#  [Pos, Neg] : shape [[?, num_domains, 1],[?, num_negative_samples,num_domains,1]]
#  train_model( model, inputs, outputs ):
# =====================================
def model_train(
        model_obj,
        pos_x,
        neg_x,
        batch_size=512,
        num_epochs=100
):
    num_samples = pos_x.shape[0]
    y = [None] * num_samples
    model_obj.fit(
        [pos_x, neg_x],
        y,
        batch_size=batch_size,
        epochs=num_epochs
    )
    return model_obj


# =========================================== #


