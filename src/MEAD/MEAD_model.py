import operator
import pickle
import math
import tensorflow as tf
import numpy as np
import glob
import pandas as pd
import os
from sklearn.manifold import TSNE

from numpy.core._multiarray_umath import ndarray
from sklearn.manifold import TSNE
import sys
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import time
import inspect
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import yaml
from tensorflow.contrib.tensorboard.plugins import projector
tf.random.set_random_seed(729)

class model:
    def __init__(self, MODEL_NAME, SAVE_DIR, OP_DIR):

        self.inference = False
        self.save_dir = SAVE_DIR
        self.op_dir = OP_DIR
        self.frozen_file = None
        self.ts = None
        self.use_bias = True
        self.save_loss_fig = True
        self.show_figure = False
        self._model_name = MODEL_NAME
        self.num_neg_samples = 3
        self.lambda_1 = 1
        self.lambda_2 = 1
        self.lambda_3 = 1
        self._enable_l2_loss = True

        return

    def set_model_hyperparams(
            self,
            domain_dims,
            emb_dims,
            use_bias=True,
            batch_size=128,
            num_epochs=20,
            learning_rate=0.001,
            num_neg_samples=3
    ):

        MODEL_NAME = self._model_name
        self.learning_rate = learning_rate
        self.num_domains = len(domain_dims)
        self.domain_dims = domain_dims
        self.num_emb_layers = len(emb_dims)
        self.emb_dims = emb_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_signature = MODEL_NAME + '_'.join([str(e) for e in emb_dims])
        self.use_bias = use_bias
        self.num_neg_samples = num_neg_samples
        self.summary_data_loc = os.path.join(
            self.op_dir,
            'summary_data'
        )
        if not os.path.exists(self.summary_data_loc):
            os.mkdir(self.summary_data_loc)

        return

    def set_model_options(
            self,
            show_loss_figure,
            save_loss_figure
    ):
        self.show_loss_figure = show_loss_figure
        self.save_loss_figure = save_loss_figure
        self.set_w_mean = True
        self.inference = False
        return

    def set_l2_loss_flag(self, flag=False):
        self._enable_l2_loss = flag

    def get_weight_variable(
            self,
            shape,
            name=None
    ):

        initializer = tf.contrib.layers.xavier_initializer()
        if name is not None:
            return tf.Variable(initializer(shape), name=name)
        else:
            return tf.Variable(initializer(shape))

    def define_wbs(self):
        # print('>> Defining weights :: start')

        self.W = [None] * self.num_emb_layers
        self.b = [None] * self.num_emb_layers

        wb_scope_name = 'params'
        # doman dimensions

        layer_1_dims = []
        for i in self.domain_dims:
            _d = int(math.ceil(math.log(i, 2)))
            if _d <= 1:
                _d += 1
            layer_1_dims.append(_d)
        # print(layer_1_dims)

        with tf.name_scope(wb_scope_name):
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            self.wb_names = []

            # -------
            # For each layer define W , b
            # -------
            for l in range(self.num_emb_layers):
                self.W[l] = [None] * self.num_domains
                self.b[l] = [None] * self.num_domains

                # print("----> Layer", (l + 1))
                if l == 0 :
                    layer_inp_dims = self.domain_dims
                    layer_op_dims = layer_1_dims
                    layer_op_dims = self.emb_dims[0]

                else:
                    if l == 1 :
                        layer_inp_dims = layer_1_dims
                    else:
                        layer_inp_dims = [self.emb_dims[l - 1]] * self.num_domains
                    layer_op_dims = [self.emb_dims[l]] * self.num_domains

                for d in range(self.num_domains):

                    _name = 'W_layer_' + str(l) + '_domain_' + str(d)

                    if self.inference is True:
                        n = prefix + _name + ':0'
                        self.W[l][d] = self.restore_graph.get_tensor_by_name(n)
                    else:

                        z = self.get_weight_variable(
                            [layer_inp_dims[d],
                             layer_op_dims],
                            name=_name)
                        self.W[l][d] = z
                        self.wb_names.append(prefix + _name)

                if self.use_bias:
                    for d in range(self.num_domains):
                        _name_b = 'bias_layer_' + str(l) + '_domain_' + str(d)
                        b_dims = [layer_op_dims]  # opdim1, opdim 2

                        if self.inference is True:
                            n = prefix + _name_b + ':0'
                            self.b[l][d] = self.restore_graph.get_tensor_by_name(n)
                        else:
                            z = self.get_weight_variable(b_dims, _name_b)
                            self.b[l][d] = z
                            self.wb_names.append(prefix + _name_b)

            '''
            define weights
            for each item : for norm calculation
            '''
            self.W_item = [None] * self.num_domains

            for d in range(self.num_domains):
                _name = 'W_item_domain_' + str(d)
                if self.inference is True:
                    n = prefix + _name + ':0'
                    self.W_item[d] = self.restore_graph.get_tensor_by_name(n)
                else:
                    z = self.get_weight_variable(
                        [self.emb_dims[-1]],
                        name=_name
                    )
                    self.W_item[d] = z
                    self.wb_names.append(prefix + _name)

        # print('>> Defining weights :: end')

    # --------------------------- #

    def restore_model(self):

        # Model already restored!
        if self.inference is True:
            return

        self.inference = True

        if self.frozen_file is None:
            # ensure embedding dimensions are correct
            emb = '_'.join([str(_) for _ in self.emb_dims])
            files = glob.glob(os.path.join(self.save_dir, 'checkpoints', '*' + emb + '*.pb'))
            f_name = files[-1]
            self.frozen_file = f_name

        if self.ts is None:
            self.ts = '.'.join(
                (''.join(
                    self.frozen_file.split('_')[-1]
                )
                ).split('.')[:1])
        # print('ts ::', self.ts)

        tf.reset_default_graph()
        # print('Frozen file', self.frozen_file)

        with tf.gfile.GFile(self.frozen_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.restore_graph = None

        with tf.Graph().as_default() as g:
            try:
                tf.graph_util.import_graph_def(
                    graph_def,
                    input_map=None,
                    name='',
                    return_elements=None,
                    op_dict=None,
                    producer_op_list=None
                )
            except:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    name='',
                    return_elements=None,
                    op_dict=None,
                    producer_op_list=None
                )
            self.restore_graph = g
            self.inference = True
            self.build_model()
        return

    # ---------------------------------------------------------- #

    def _add_var_summaries(self):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
            continue
            #
            # with tf.name_scope('summaries'):
            #     mean = tf.reduce_mean(var)
            #     tf.summary.scalar('mean', mean)
            #
            # with tf.name_scope('stddev'):
            #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            #     tf.summary.scalar('stddev', stddev)
            #     tf.summary.scalar('max', tf.reduce_max(var))
            #     tf.summary.scalar('min', tf.reduce_min(var))
        return

    def get_inp_embeddings(self, x_inp):

        x_WXb = [None] * self.num_domains
        for d in range(self.num_domains):
            # for each domain
            prev = None
            for l in range(self.num_emb_layers):

                if l == 0:
                    a = tf.nn.embedding_lookup(
                        self.W[l][d],
                        x_inp[d]
                    )
                    _wx = tf.squeeze(a, axis=1)

                else:
                    _x = prev
                    _wx = tf.matmul(
                        _x,
                        self.W[l][d]
                    )

                if self.use_bias:
                    _wx_b = tf.add(_wx, self.b[l][d])
                else:
                    _wx_b = _wx

                prev = _wx_b
            x_WXb[d] = prev
        return x_WXb

    def build_model(self):

        # print('Building model : start ')
        self.model_scope_name = 'model'

        with tf.variable_scope(self.model_scope_name):
            # batch_size ,domains, label_id

            self.x_pos_inp = tf.compat.v1.placeholder(
                tf.int32,
                [None, self.num_domains]
            )

            # Inside the scope	- define the weights and biases
            self.define_wbs()
            x_pos_inp = tf.split(
                self.x_pos_inp,
                self.num_domains,
                axis=-1
            )

            emb_op_pos = self.get_inp_embeddings(x_pos_inp)
            # print(' emb_op_pos shape ', emb_op_pos[0].shape)

            self.joint_emb_op_arr = emb_op_pos

            '''
            Calculate weighted norm
            '''

            self.r_b = self.get_norm_b(self.joint_emb_op_arr)
            # print(' > self.r_b ', self.r_b.shape)

            '''
            Optimization stage
            '''

            # small epsilon to avoid any sort of underflows
            _epsilon = tf.constant(math.pow(10.0, -7.0))

            # self.pairwise_ang_dist = self.calculate_angular_sim(emb_op_pos)

            if self.inference is False:
                self.neg_sample_optimization()
            else:
                return

            norm_loss = []

            for i in range(self.num_domains):
                _a = emb_op_pos[i]
                _b = self.W_item[i]
                _ab = tf.multiply(_a, _b)

                t1 = tf.norm(_ab, axis=-1, keep_dims=True)
                t1 = tf.square(1 - t1)
                norm_loss.append(t1)

            norm_loss = tf.stack(norm_loss, axis=1)
            norm_loss = tf.squeeze(norm_loss, axis=-1)
            norm_loss = tf.reduce_mean(norm_loss, axis=-1, keepdims=True)
            # print(norm_loss.shape)

            self.loss_2 = norm_loss

            self.loss =  self.loss_3 + self.loss_2
            # print(' shape of loss -->> ', self.loss.shape)
            self._enable_l2_loss = False
            # L2 regularization of weights in embedding layer
            # print('L2 loss flag ', self._enable_l2_loss)
            if  self._enable_l2_loss:
                regularizer_beta = tf.constant(math.pow(10, -2))
                for _W in self.W_item:
                    regularizer = [regularizer_beta * tf.nn.l2_loss(_W)]
                    for _ in regularizer:
                        self.loss += _

            # print(' shape of loss -->> ', self.loss.shape)

            tf.summary.scalar('loss', tf.reduce_mean(self.loss))

            self._add_var_summaries()
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )

            gvs = self.optimizer.compute_gradients(self.loss)

            tf.summary.merge(
                [tf.summary.histogram(
                    "%s-grad" % g[1].name, g[0]
                ) for g in gvs if g[0] is not None
                ]
            )

            self.gradients = gvs
            tf.summary.all_v2_summary_ops()
            capped_gvs = [
                (tf.clip_by_value(grad, -1.0, 1.0), var)
                for grad, var in gvs if grad is not None
            ]

            self.train_opt = self.optimizer.apply_gradients(capped_gvs)
            self.summary = tf.summary.merge_all()
            return

    '''
    batch_d_arr_tensor :  [[?,dom1] , [?,dom_2], ... ]
    '''

    def get_norm_b(self, batch_d_arr_tensor, reciprocal=False):
        norm_b = []

        for d in range(self.num_domains):
            _a = batch_d_arr_tensor[d]
            _b = self.W_item[d]
            _ab =  tf.multiply(_a, _b)
            norm_b.append(_ab)

        norm_b = tf.stack(norm_b, axis=1)
        self.weighted_emb = norm_b
        norm_b = tf.reduce_sum(norm_b, axis=-2)

        norm_b = tf.reduce_sum(tf.square(norm_b), axis=-1, keepdims=True)

        if reciprocal:
            norm_b = tf.pow(norm_b, -1)
        r_b = tf.tanh( norm_b )
        return r_b

    # This part should be used for norm based optimization
    def neg_sample_optimization(self):

        # ---------------
        # batch_size, domains, label_id
        # ---------------
        self.x_neg_inp = tf.compat.v1.placeholder(
            tf.int32, [
                None,
                self.num_neg_samples,
                self.num_domains
            ]
        )
        # Split
        x_neg_inp_arr = tf.split(
            self.x_neg_inp,
            num_or_size_splits=self.num_neg_samples,
            axis=1
        )

        '''
        Create list of N (= num of neg samples) list of inputs of shape [?, num_domains]
        '''
        x_neg_inp_arr = [tf.squeeze(_, axis=1) for _ in x_neg_inp_arr]

        r_b_neg = []


        for _neg in range(self.num_neg_samples):
            x_inp_neg = tf.split(
                x_neg_inp_arr[_neg],
                self.num_domains,
                axis=-1
            )
            # Get the embedding of negative sample
            emb_op_n = self.get_inp_embeddings(x_inp_neg)

            r_b = self.get_norm_b(
                batch_d_arr_tensor=emb_op_n,
                reciprocal=True
            )
            r_b_neg.append(r_b)

        r_b_neg = tf.stack(
            r_b_neg,
            axis=-1
        )

        r_b_neg = tf.squeeze(
            r_b_neg,
            axis=1
        )

        # print('r_b_neg shape ', r_b_neg.shape)

        # calculate loss
        loss_3 = tf.log(self.r_b)
        _tmp = tf.log(r_b_neg)
        _tmp = tf.reduce_sum(
            _tmp,
            axis=-1,
            keepdims=True
        )

        self.loss_3 = -tf.add(loss_3, _tmp)
        tf.summary.scalar('loss_3', tf.reduce_mean(self.loss_3))
        return

    def set_pretrained_model_file(self, f_path):
        self.frozen_file = f_path
        return

    def train_model(self, x_pos, x_neg):
        # print('Start of training :: ')
        self.ts = str(time.time()).split('.')[0]
        f_name = 'frozen' + '_' + self.model_signature + '_' + self.ts + '.pb'

        self.frozen_file = os.path.join(
            self.save_dir, 'checkpoints', f_name
        )

        self.sess = tf.compat.v1.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()
        bs = self.batch_size
        x_pos = x_pos

        num_batches = x_pos.shape[0] // bs
        losses = []

        Check_Save_Prev = False

        # print('Num batches :', num_batches)

        summary_writer = tf.summary.FileWriter(self.summary_data_loc)
        step = 0

        '''
        implement early stopping based on loss 3 
        '''

        for e in range(self.num_epochs):

            # print('epoch :: ', e)
            t1 = time.time()
            _loss_1 = []
            _loss_3 = []

            for _b in range(num_batches):

                _x_pos = x_pos[_b * bs: (_b + 1) * bs]
                _x_neg = x_neg[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x_pos = x_pos[_b * bs:]
                    _x_neg = x_neg[_b * bs:]

                if _b == 0:
                    print(_x_pos.shape)
                    print(_x_neg.shape)
                    
                _, summary,  loss = self.sess.run(
                    [self.train_opt, self.summary, self.loss],
                    feed_dict = {
                        self.x_pos_inp: _x_pos,
                        self.x_neg_inp: _x_neg
                    }
                )

                batch_loss = np.mean(loss)
                losses.append(batch_loss)

                # if _b % 100 == 0 :
                #     print(' batch ::', _b)
                #     print(batch_loss)

                # summary_writer.add_summary(summary, step)
                step += 1

                if np.isnan(batch_loss):
                    Check_Save_Prev = True
                    print('[ERROR] Loss is NaN !!!, breaking...')
                    break

            '''
            check the direction of pairwise angular distance
            '''


            if Check_Save_Prev is True:
                break
            if e == self.num_epochs-1:
                graph_def = tf.get_default_graph().as_graph_def()
                frozen_graph_def = convert_variables_to_constants(
                    self.sess,
                    graph_def,
                    self.wb_names
                )
                with tf.gfile.GFile(self.frozen_file, "wb") as f:
                    f.write(frozen_graph_def.SerializeToString())

                # t2 = time.time()
                # t = (t2 - t1) / 60
                # print('Epoch ', e + 1, 'Time elapsed in epoch : ', t, 'minutes')

        if self.save_loss_fig or self.show_loss_figure:
            plt.figure()
            plt.title('Training Loss')
            plt.xlabel('batch')
            plt.ylabel('loss')
            plt.plot(range(len(losses)), losses, 'r-')

            if self.save_loss_figure:
                fig_name = 'loss_' + self.model_signature + '_epochs_' + str(self.num_epochs) + '_' + self.ts + '.png'
                file_path = os.path.join(self.op_dir, fig_name)
                plt.savefig(file_path)

            if self.show_loss_figure:
                plt.show()
            plt.close()
        return self.frozen_file


    def get_event_score(
            self,
            x
    ):
        self.restore_model()
        output = []
        bs = self.batch_size
        num_batches = x.shape[0] // bs

        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches):
                _x = x[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x = x[_b * bs:]
                _output = sess.run(
                    self.r_b,
                    feed_dict={
                        self.x_pos_inp: _x,
                    }
                )

                output.extend(_output)
            res = np.array(output)
            return res

    '''
    Get just the embeddings
    '''

    def get_record_embeddings(
            self,
            x
    ):
        self.restore_model()
        output = []
        bs = self.batch_size
        num_batches = x.shape[0] // bs

        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches):
                _x = x[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x = x[_b * bs:]
                _output = sess.run(
                    self.weighted_emb,
                    feed_dict={
                        self.x_pos_inp: _x,
                    }
                )

                output.extend(_output)
            res = np.array(output)
            return res


