import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys
import glob
import math
import matplotlib.pyplot as plt
import time
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.contrib.tensorboard.plugins import projector
tf.random.set_random_seed(729)

class model:

    def __init__(self, MODEL_NAME, SAVE_DIR, OP_DIR):
        self.inference = False
        self.save_dir = SAVE_DIR
        self.op_dir = OP_DIR
        self.frozen_file = None
        self.ts = None
        self.save_loss_fig = True
        self.show_figure = False
        self._model_name = MODEL_NAME
        self.num_neg_samples = 3
        self._epsilon = 0.0001
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
        self.latent_dim = emb_dims[-1]
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_signature = MODEL_NAME + '_'.join([str(e) for e in emb_dims])
        self.use_bias = use_bias
        self.num_neg_samples = num_neg_samples
        self.summary_data_loc = os.path.join(
            self.op_dir,
            'summary_data'
        )
        if not os.path.exists(self.op_dir):
            os.mkdir(self.op_dir)
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
        # domain dimensions

        layer_1_dims = []
        for i in self.domain_dims:
            _d = int(math.ceil(math.log(i, 2)))
            if _d <= 1:
                _d += 1
            layer_1_dims.append(_d)

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
                        b_dims = [layer_op_dims]  # opdim 1, opdim 2

                        if self.inference is True:
                            n = prefix + _name_b + ':0'
                            self.b[l][d] = self.restore_graph.get_tensor_by_name(n)
                        else:
                            z = self.get_weight_variable(b_dims, _name_b)
                            self.b[l][d] = z
                            self.wb_names.append(prefix + _name_b)


            self.interaction_W = {}

            for i in range(self.num_domains):
                self.interaction_W[i] = {}
                for j in range(i+1,self.num_domains):
                    _name = 'int_W_' + str(i) + str(j)

                    if self.inference is True:
                        n = prefix + _name + ':0'
                        self.interaction_W[i][j] = self.restore_graph.get_tensor_by_name(n)
                    else:
                        _shape = [self.latent_dim,1]
                        z = self.get_weight_variable(
                            _shape,
                            name=_name)
                        self.interaction_W[i][j] = z
                        self.wb_names.append(prefix + _name)

            _name = 'conv_kernel'

            if self.inference is True:
                n = prefix + _name + ':0'
                self.conv_kernel = self.restore_graph.get_tensor_by_name(n)
            else:
                k = int(self.num_domains* (self.num_domains-1)/2)
                _shape = [k, 1, 1, 1]
                self.conv_kernel = self.get_weight_variable(
                    _shape,
                    name=_name
                )

            _name = 'W_final'

            if self.inference is True:
                n = prefix + _name + ':0'
                self.W_final = self.restore_graph.get_tensor_by_name(n)
            else:
                k = int(self.num_domains * (self.num_domains - 1) / 2)
                _shape = [self.latent_dim,1]
                self.W_final = self.get_weight_variable(
                    _shape,
                    name=_name
                )

        return

    def calculate_interactions(
            self,
            _input
    ):
        res = []
        for i in range(self.num_domains):
            for j in range(i+1,self.num_domains):
                a = _input[i]
                a = tf.expand_dims(a,axis=-1)
                b = _input[j]
                b = tf.expand_dims(b,axis=-1)

                print(a.shape,b.shape)
                _w = tf.transpose(
                    self.interaction_W[i][j],
                    perm = [1,0]
                )

                c = tf.einsum('ijk,kl->ijl', a, _w)
                print(c.shape)

                d = tf.einsum('ijk,ikl->ijl', c, b)
                print(d.shape)

                res.append(d)
        res = tf.stack(res,axis=1)
        print(' Shape before conv ', res.shape)


        res = tf.nn.conv2d(
            input = res,
            filter = self.conv_kernel,
            strides = [1,1,1,1],
            padding = 'VALID'
        )
        res = tf.squeeze(res, axis=-1)
        res = tf.squeeze(res, axis=1)
        return res

    def _add_var_summaries(self):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

    # input is [ [batch , 1], [batch,1] ... #domains ]
    def get_inp_embeddings(
            self,
            x_inp
    ):

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
            print(x_WXb[d].shape)

        return x_WXb

    def get_tensor_score(self, _tensor, neg_sample = False ):
        res = tf.matmul(_tensor,self.W_final)
        if neg_sample:
            res = tf.math.reciprocal(res)
        res = tf.exp(res)
        res = tf.tanh(res)
        return res

    def neg_sample_optimization(self):
        # ---------------
        # batch_size, domains, label_id
        # ---------------
        self.x_neg_inp = tf.placeholder(
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

        neg_res = []
        for _neg in x_neg_inp_arr:

            _neg = tf.squeeze(_neg,axis=1)
            _neg = tf.split(
                _neg,
                num_or_size_splits=self.num_domains,
                axis=1
            )
            _neg = self.get_inp_embeddings(_neg)
            _neg = self.calculate_interactions(_neg)
            _neg = self.get_tensor_score(_neg,True)
            neg_res.append(_neg)

        neg_res = tf.stack(neg_res, axis=1)
        neg_res = tf.log(neg_res)
        neg_res = tf.math.reduce_mean( neg_res, axis=1 , keepdims= False)
        print(' neg_res >> ', neg_res.shape)
        return neg_res

    def build_model(self):
        self.model_scope_name = 'model'
        with tf.variable_scope(self.model_scope_name):
            # batch_size ,domains, label_id
            self.x_pos_inp = tf.placeholder(
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

            x_WXb = self.get_inp_embeddings(x_pos_inp)
            _tensor = self.calculate_interactions(x_WXb)
            val1 = self.get_tensor_score(_tensor)
            if self.inference :
                return
            else:
                val1 = tf.log(val1 + self._epsilon)
                val2 = self.neg_sample_optimization()
                self.loss =  - (val1 + val2)

            self._add_var_summaries()
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )

            self.train_opt = self.optimizer.minimize(
                self.loss
            )

        return


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

        tf.reset_default_graph()

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

    def train_model(self, x_pos, x_neg):
        print('Start of training :: ')
        self.ts = str(time.time()).split('.')[0]
        f_name = 'frozen' + '_' + self.model_signature + '_' + self.ts + '.pb'

        self.frozen_file = os.path.join(
            self.save_dir, 'checkpoints', f_name
        )

        self.sess = tf.InteractiveSession()
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
            print('epoch :: ', e)
            t1 = time.time()

            for _b in range(num_batches):
                _x_pos = x_pos[_b * bs: (_b + 1) * bs]
                _x_neg = x_neg[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x_pos = x_pos[_b * bs:]
                    _x_neg = x_neg[_b * bs:]

                if _b == 0:
                    print(_x_pos.shape)

                _, summary,  loss = self.sess.run(
                    [self.train_opt, self.summary, self.loss],
                    feed_dict={
                        self.x_pos_inp: _x_pos,
                        self.x_neg_inp: _x_neg
                    }
                )

                batch_loss = np.mean(loss)
                losses.append(batch_loss)

                if _b % 100 == 0 :
                    print(' batch ::', _b)
                    print(batch_loss)

                summary_writer.add_summary(summary, step)
                step += 1

                if np.isnan(batch_loss):
                    Check_Save_Prev = True
                    print('[ERROR] Loss is NaN !!!, breaking...')
                    break

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

                t2 = time.time()
                t = (t2 - t1) / 60
                print('Epoch ', e + 1, 'Time elapsed in epoch : ', t, 'minutes')

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





obj = model('trial','save_dir','op_dir')

obj.set_model_hyperparams(
    domain_dims=[25,40,75,150],
            emb_dims=[12],
            use_bias=True,
            batch_size=128,
            num_epochs=20,
            learning_rate=0.001,
            num_neg_samples=5
)
obj.build_model()


