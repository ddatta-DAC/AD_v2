import torch
import torch.nn as nn
import logging
from torch import FloatTensor as FT
from torch import LongTensor as LT

try:
    from src.Classifiers.wide_n_deep_model import clf_wide_n_deep as clf_WIDE_N_DEEP
    from src.Classifiers.deepFM import clf_deepFM  as clf_DEEP_FM
    from clf_net import clf_net_v2 as clf_MLP
    from record_node import graph_net_v2 as graph_net
    from gam_module import agreement_net_v2 as gam_net

except:
    from .src.Classifiers.wide_n_deep_model import clf_wide_n_deep as clf_WIDE_N_DEEP
    from .src.Classifiers.deepFM import clf_deepFM  as clf_DEEP_FM
    from .clf_net import clf_net_v2 as clf_MLP
    from .record_node import graph_net_v2 as graph_net
    from .gam_module import agreement_net_v2 as gam_net


# =========================================
# Main module that encapsulates everything.
# =========================================
class SS_network(nn.Module):
    def __init__(
            self,
            DEVICE,
            node_emb_dimension,
            num_domains,
            matrix_pretrained_node_embeddings,  # [Number of entities, embedding dimension]
            list_gam_encoder_dimensions=None,
            clf_type=None,
            dict_clf_initilize_inputs=None  # A dict of of parameters required to initialize each specific classifier
    ):
        super(SS_network, self).__init__()
        # valid values for train_mode are 'f', 'g', False
        self.train_mode = False
        self.test_mode = False
        self.clf_type = clf_type

        self.agreement_net = None
        self.clf_net = None
        self.graph_net = graph_net(
            node_emb_dimension,
            num_domains,
            matrix_pretrained_node_embeddings
        )

        gam_record_input_dimension = node_emb_dimension * num_domains
        self.agreement_net = gam_net(
            gam_record_input_dimension,
            list_gam_encoder_dimensions,
            activation='selu'
        )

        if clf_type == 'wide_n_deep':

            self.clf_net = clf_WIDE_N_DEEP(
                wide_inp_01_dim=dict_clf_initilize_inputs['wide_inp_01_dim'],
                pretrained_node_embeddings=matrix_pretrained_node_embeddings,  # type : FloatTensor,
                tune_entity_emb=dict_clf_initilize_inputs['tune_entity_emb'],
                deep_FC_layer_dims=dict_clf_initilize_inputs['deep_FC_layer_dims'],
                num_entities=None,  # total number of entities (reqd if no pretrained emb)
                entity_emb_dim=node_emb_dimension,
                num_domains=num_domains
            )
        elif clf_type == 'deepFM':
            self.clf_net = clf_DEEP_FM(
                wide_inp_01_dim=dict_clf_initilize_inputs['wide_inp_01_dim'],  # the size of the 1-0 vector
                num_domains=num_domains,  # number of fields
                entity_emb_dimensions=node_emb_dimension,  # emb dimension of each entity
                FM_inp_dim=None,
                dnn_layer_dimensions=dict_clf_initilize_inputs['dnn_layer_dimensions'],
                # FNN for concatenated embedding
                pretrained_node_embeddings=matrix_pretrained_node_embeddings,  # type : FloatTensor,
                tune_entity_emb=dict_clf_initilize_inputs['tune_entity_emb'],
                num_entities=None
            )
        elif clf_type == 'MLP':
            self.clf_net = clf_MLP(
                input_dim=node_emb_dimension * num_domains,
                mlp_layer_dims=dict_clf_initilize_inputs['mlp_layer_dims'],
                dropout=dict_clf_initilize_inputs['dropout'],
                activation=dict_clf_initilize_inputs['activation']
            )

        self.clf_net.to(DEVICE)
        self.agreement_net.to(DEVICE)
        self.graph_net.to(DEVICE)

        return

    # ---------------------------
    # Input should be [ Batch, record( list of entities ) ]
    # record( list of entities ) should have serialized entity id
    # ---------------------------
    # input_xy is an list
    def forward(
            self, input_x, input_y=None
    ):
        # ----------------
        # Train the agreement module
        # ----------------
        if self.train_mode == 'g':
            x1 = input_x[0]
            x2 = input_x[1]
            # print('[Forward] g ; shapes of x1 and x2 :', x1.shape, x2.shape)
            x1 = self.graph_net(x1)
            x2 = self.graph_net(x2)

            y_pred = self.agreement_net(
                x1,
                x2
            )
            return y_pred

        elif self.train_mode == 'f':
            x1 = input_x
            x1 = self.graph_net(x1)
            y_pred = self.clf_net(x1)
            return y_pred

        elif self.train_mode == 'f_ll':
            x1_G = input_x[0]
            x2_G = input_x[1]
            x1_F = input_x[2]

            x1 = self.graph_net(x1_G)
            x2 = self.graph_net(x2_G)
            pred_agreement = self.agreement_net(x1, x2)
            if self.clf_type == 'MLP':
                x1_F = self.graph_net(x1_F)

            pred_y1 = self.clf_net(x1_F)
            return pred_agreement, pred_y1

        elif self.train_mode == 'f_ul':

            x1_G = input_x[0]
            x2_G = input_x[1]
            x1_F = input_x[2]

            if self.clf_type == 'MLP':
                x1_F = self.graph_net(x1_F)
            pred_y1 = self.clf_net(x1_F)

            x1 = self.graph_net(x1_G)
            x2 = self.graph_net(x2_G)
            pred_agreement = self.agreement_net(x1, x2)
            return pred_agreement, pred_y1

        elif self.train_mode == 'f_uu':

            x1_G = input_x[0]
            x2_G = input_x[1]
            x1_F = input_x[2]
            x2_F = input_x[3]

            x1 = self.graph_net(x1_G)
            x2 = self.graph_net(x2_G)
            pred_agreement = self.agreement_net(x1, x2)
            if self.clf_type == 'MLP':
                x1_F = self.graph_net(x1_F)
                x2_F = self.graph_net(x2_F)

            pred_y1 = self.clf_net(x1_F)
            pred_y2 = self.clf_net(x2_F)
            return pred_agreement, pred_y1, pred_y2

        if self.test_mode == True:
            x1 = input_x
            x1 = self.graph_net(x1)
            y_pred = self.clf_net(x1)
            return y_pred
