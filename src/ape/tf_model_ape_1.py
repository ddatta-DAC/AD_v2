import pickle
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import inspect
import matplotlib.pyplot as plt
import sys
import yaml
import time
from collections import OrderedDict

import glob

tf.set_random_seed(729)

OP_DIR = None
_DIR = None


class model_ape_1:
    def __init__(self, model_name):
        self.model_name = model_name
        return

    def set_hyper_parameters(
            self,
            emb_dims,
            use_bias
    ):
        self.emb_dims = emb_dims
        self.num_layers = len(emb_dims)
        self.use_bias = use_bias
        self.epsilon = 0.000001
        self.emb_str = '_'.join([str(_) for _ in self.emb_dims])
        f_name = self.model_name + '_' + self.emb_str + '_k_' + str(self.neg_samples) + "_frozen.pb"
        self.frozen_filename = os.path.join(self.chkpt_dir, f_name)
        print(self.frozen_filename)

    def set_model_params(
            self,
            num_entities,
            neg_samples,
            inp_dims,
            batch_size=32,
            num_epochs=5,
            lr=0.001,
            chkpt_dir=None
    ):
        global MODEL_NAME
        self.learning_rate = lr
        self.neg_samples = neg_samples
        self.num_entities = num_entities
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.inp_dims = inp_dims
        self.chkpt_dir = chkpt_dir

    # -------- Restore ---------- #
    def restore_model(self):
        tf.reset_default_graph()

        with tf.io.gfile.GFile(self.frozen_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        self.restore_graph = None

        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(
                graph_def,
                input_map=None,
                name='',
                return_elements=None,
                op_dict=None,
                producer_op_list=None
            )
            self.restore_graph = g
            self.build_model(
                inference=True
            )
        return

    # ------- Define weights ------ #
    def get_weight_variable(self, shape, name=None):
        init_val = None
        try:
            init_val = tf.random_uniform(shape, -0.10, 0.10)
        except:
            init_val = tf.random.uniform(shape, -0.10, 0.10)

        if name is not None:
            return tf.Variable(
                init_val,
                name=name
            )
        else:
            return tf.Variable(
                init_val
            )

    def define_wbs(self):
        # print('Defining weights :: start')
        self.W = [None] * self.num_layers
        self.B = [None] * self.num_layers

        for l in range(self.num_layers):
            W_l = [None] * self.num_entities
            self.W[l] = W_l

            # Only if bias is being used in that layer
            if self.use_bias[l] is True:
                B_l = [None] * self.num_entities
                self.B[l] = B_l

        # store names to restore graph
        self.wb_names = []

        wb_scope_name = 'params'
        _inp_dim = None
        self.W = [None] * self.num_layers
        self.B = [None] * self.num_layers

        with tf.name_scope(wb_scope_name):
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            # Declare/Define weights
            for l in range(self.num_layers):
                self.W[l] = [None] * self.num_entities
                self.B[l] = [None] * self.num_entities

                for i in range(self.num_entities):
                    if l == 0:
                        _inp_dim = self.inp_dims[i]
                    else:
                        _inp_dim = self.emb_dims[l - 1]

                    w_dims = [_inp_dim, self.emb_dims[l]]
                    name = 'W_' + str(l) + str(i)
                    if self.restore is True:
                        n = prefix + name + ':0'
                        self.W[l][i] = self.restore_graph.get_tensor_by_name(n)
                    else:
                        z = self.get_weight_variable(w_dims, name)
                        self.W[l][i] = z
                    self.wb_names.append(prefix + name)

                    # Only if bias is being used for the layer #
                    if self.use_bias[l] is True:
                        name = 'B_' + str(l) + str(i)
                        b_dims = [self.emb_dims[l]]
                        if self.restore is True:
                            n = prefix + name + ':0'
                            self.B[l][i] = self.restore_graph.get_tensor_by_name(n)
                        else:
                            z = self.get_weight_variable(b_dims, name)
                            self.B[l][i] = z
                        self.wb_names.append(prefix + name)

            # print('------')

            name = 'c'
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            if self.restore is False:
                init_val = None
                try:
                    init_val = tf.random_uniform([1], 0, 1)
                except:
                    init_val = tf.random.uniform([1], 0, 1)
                self.c = tf.Variable(
                    initial_value=init_val,
                    name=name
                )
                self.wb_names.append(prefix + name)
            else:
                n = prefix + name + ':0'
                self.c = self.restore_graph.get_tensor_by_name(n)
            # print(self.c)

            name = 'W_ij'
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            if self.restore is False:
                init_val = None
                try:
                    init_val = tf.random.uniform(
                        [self.num_entities, self.num_entities], 0, 1)
                except:
                    init_val = tf.random_uniform(
                        [self.num_entities, self.num_entities], 0, 1)

                self.W_ij = tf.Variable(
                    init_val,
                    name=name
                )
                self.wb_names.append(prefix + name)
            else:
                n = prefix + name + ':0'
                self.W_ij = self.restore_graph.get_tensor_by_name(n)

            # print(self.W_ij)
            # print('Defining weights :: end')

        return

    # inference = true -> inference stage, Use pre-trained model
    # inference_case valid only for inference=True
    # Note this is called after restoring model as well

    def build_model(
            self,
            inference=False
    ):
        # Set placeholders

        self.restore = inference
        # Input indices
        # print('Building model : start ')
        self.output_node_names = []
        self.model_scope_name = 'model'

        with tf.variable_scope(self.model_scope_name):
            self.x_pos_inp = tf.placeholder(
                tf.int32,
                [None, self.num_entities],
                name='x_pos_ev'
            )

            # -------------------- #
            # The model already trained , no need for negative samples
            if inference is False:
                # Negative sample
                self.x_neg_inp = tf.placeholder(
                    tf.int32,
                    [None, self.neg_samples, self.num_entities],
                    name='x_neg_samples'
                )

                # Placeholder for probabilities of each entity in each domain
                # Each entity_domain has own list
                self.term_2 = tf.placeholder(
                    tf.float32,
                    shape=([None, 1])
                )

                self.term_4 = tf.placeholder(
                    tf.float32,
                    shape=(
                        [None, self.neg_samples, 1]
                    )
                )

            # -------------------- #

            self.define_wbs()

            x_pos_inp = tf.split(
                self.x_pos_inp,
                self.num_entities,
                axis=-1
            )

            for i in range(self.num_entities):
                z_i = x_pos_inp[i]
                z_i = tf.one_hot(
                    axis=-1,
                    indices=z_i,
                    depth=self.inp_dims[i]
                )
                z_i = tf.squeeze(z_i, axis=1)
                x_pos_inp[i] = z_i

            # print('Shape of inputs , after one hot encoding')
            # print([k.shape.as_list() for k in x_pos_inp])

            if inference is False:
                x_neg_inp = tf.split(
                    self.x_neg_inp,
                    self.num_entities,
                    axis=-1
                )

                for i in range(self.num_entities):
                    z_i = x_neg_inp[i]
                    z_i = tf.one_hot(
                        axis=-1,
                        indices=z_i,
                        depth=self.inp_dims[i]
                    )
                    z_i = tf.squeeze(z_i, axis=2)
                    x_neg_inp[i] = z_i

                # print('Shape of inputs , after one hot encoding')
                # print(
                #     [k.shape.as_list() for k in x_neg_inp]
                # )
            # print('-----')

            x_pos_WXb = [None] * self.num_layers

            # WX + B
            # for positive samples
            for l in range(0, 1):
                x_pos_WXb[l] = [None] * self.num_entities
                for i in range(self.num_entities):
                    if l == 0:
                        _x = x_pos_inp[i]
                    else:
                        _x = x_pos_WXb[l - 1][i]
                    _wx = tf.matmul(_x, self.W[l][i])
                    if self.use_bias[l]:
                        _wx_b = tf.add(_wx, self.B[l][i])
                    else:
                        _wx_b = _wx
                    # print(_wx_b)
                    x_pos_WXb[l][i] = _wx_b

            # ---------------- #
            # WX + B
            # for negative samples

            if inference is False:
                x_neg_WXb = [None] * self.num_layers
                for l in range(0, 1):
                    x_neg_WXb[l] = [None] * self.num_entities
                    for i in range(self.num_entities):
                        if l == 0:
                            _x = x_neg_inp[i]
                        else:
                            _x = x_neg_WXb[l - 1][i]
                        # print(self.W[l][i])
                        _wx = tf.einsum('ijk,kl -> ijl', _x, self.W[l][i])

                        if self.use_bias[l]:
                            _wx_b = tf.add(_wx, self.B[l][i])
                        else:
                            _wx_b = _wx
                        x_neg_WXb[l][i] = _wx_b
            # ---------------- #
            # x_pos_WXb has the embedding for the different entities for positive samples
            # do pairwise dot product

            _sum = 0
            for i in range(self.num_entities):
                for j in range(i + 1, self.num_entities):
                    z = tf.reduce_sum(
                        tf.multiply(
                            x_pos_WXb[l][i],
                            x_pos_WXb[l][j]
                        ),
                        axis=1,
                        keepdims=True
                    )
                    z = z * tf.square(self.W_ij[i][j])
                    _sum += z
                    # print(z)
            P_e = tf.exp(_sum + self.c)  # 1st term in the loss equation
            self.score = P_e

            # --------------------------------- #
            # End of build      -> for restore mode
            # --------------------------------- #
            if inference is True:
                return

            # ---------------- #
            # x_pos_WXb has the embedding for the different entities for negative samples
            # do pairwise dot product
            neg_pair_dp = []
            for i in range(self.num_entities):
                for j in range(i + 1, self.num_entities):
                    # print(x_neg_WXb[l][i])
                    _z1 = _wx = tf.einsum(
                        'ijk,ijk -> ijk',
                        x_neg_WXb[l][i],
                        x_neg_WXb[l][j]
                    )
                    z = tf.reduce_sum(
                        _z1,
                        axis=-1,
                        keepdims=True)
                    z = z * tf.square(self.W_ij[i][j])
                    neg_pair_dp.append(z)
            # print(neg_pair_dp)

            z1 = tf.stack(neg_pair_dp, axis=2)
            z1 = tf.squeeze(z1, axis=-1)
            z2 = tf.reduce_sum(z1, axis=-1, keepdims=False)
            z2 = z2 + self.c
            z3 = tf.exp(z2)  # 3rd term in the loss equation

            # ---------------- #

            obj = tf.log(tf.sigmoid(tf.log(P_e) - self.term_2) + self.epsilon)
            # print('1', tf.sigmoid(tf.log(P_e)))
            # print('z3', z3)
            z4 = tf.squeeze(self.term_4, axis=-1)

            z4 = tf.log(tf.sigmoid(-z3 + z4) + self.epsilon)
            z5 = tf.reduce_sum(z4, axis=-1, keepdims=True)
            obj += z5
            self.obj = obj
            # print(self.obj)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )
            self.train_opt = self.optimizer.minimize(-self.obj)

    def train_model(
            self,
            x_pos,
            x_neg,
            term_2,
            term_4
    ):
        global OP_DIR
        global _DIR
        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.compat.v1.train.Saver()
        bs = self.batch_size

        num_batches = x_pos.shape[0] // bs
        # print('-----')

        losses = []
        for e in range(self.num_epochs):
            print('Epoch :: ', e + 1)
            st = time.time()
            for _b in range(num_batches):
                                 
                _x_pos = x_pos[_b * bs: (_b + 1) * bs]
                _x_neg = x_neg[_b * bs: (_b + 1) * bs]
                _term_2 = term_2[_b * bs: (_b + 1) * bs]
                _term_4 = term_4[_b * bs: (_b + 1) * bs, :, :]

                _, loss = self.sess.run(
                    [self.train_opt, self.obj],
                    feed_dict={
                        self.x_pos_inp: _x_pos,
                        self.x_neg_inp: _x_neg,
                        self.term_2: _term_2,
                        self.term_4: _term_4
                    }
                )
                batch_loss = np.mean(loss)
                if _b % 200 == 0:
                    print('Epoch :: ', e + 1, 'Batch :', _b, ' Loss :', batch_loss)

                losses.append(batch_loss)

            ed = time.time()
            # print('Time elapsed: ', ed - st)

        # print('------------------------------->')
        losses = np.array(losses) * -1

        fig_name = 'model_ape_1_' + str(time.time()) + '.png'
        fig_path = os.path.join(OP_DIR, fig_name)
        plt.figure(figsize=[14, 10])
        plt.title('Loss for model training', fontsize=18)
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Loss value', fontsize=16)
        # sns.lineplot(x=range(len(losses)), y=losses, lw=0.75)

        plt.plot(range(len(losses)), losses, 'r-')
        plt.savefig(fig_path)
        plt.close()

        graph_def = tf.get_default_graph().as_graph_def()
        frozen_graph_def = convert_variables_to_constants(
            self.sess,
            graph_def,
            self.wb_names
        )

        with tf.io.gfile.GFile(self.frozen_filename, "wb") as f:
            print('Saving file. ', self.frozen_filename)
            f.write(frozen_graph_def.SerializeToString())
        return

    def inference(
            self,
            data,
    ):
        self.restore_model()
        output = None
        print(data.shape)
        res = []
        bs = 512
        num_batches = data.shape[0] // bs
        print(' number of batches ....', num_batches)
        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches + 1):
                _data = data[_b * bs: (_b + 1) * bs]
                output = sess.run(
                    self.score,
                    feed_dict={
                        self.x_pos_inp: _data,
                    })
                res.extend(output)
        print(len(res))
        res = np.array(res)
        return res
