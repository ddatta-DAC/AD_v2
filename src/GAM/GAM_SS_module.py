import torch
import torch.nn as nn
import logging
from torch import FloatTensor as FT
from torch import LongTensor as LT



try:
    from src.Classifiers import wide_n_deep_model as clf_WIDE_N_DEEP
    from src.Classifiers import deepFM  as clf_DEEP_FM
    from clf_net import clf_net_v2 as clf_MLP
    from record_node import graph_net_v2 as graph_net
    from  gam_module import gam_net_v1 as gam_net

except:
    from .src.Classifiers import wide_n_deep_model as clf_WIDE_N_DEEP
    from .src.Classifiers import deepFM  as clf_DEEP_FM
    from .clf_net import clf_net_v2 as clf_MLP
    from .record_node import graph_net_v2 as graph_net
    from .gam_module import gam_net_v1 as gam_net

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
            list_gam_encoder_dimensions
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

            y_pred = self.gam_net(
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
            x1 = input_x[0]
            x2 = input_x[1]
            x1 = self.graph_net(x1)
            x2 = self.graph_net(x2)

            pred_y1 = torch.argmax(
                self.clf_net(x1),
                dim=1
            )

            pred_agreement = self.gam_net(x1, x2)
            return pred_agreement, pred_y1

        elif self.train_mode == 'f_ul':
            x1 = input_x[0]
            x2 = input_x[1]
            x1 = self.graph_net(x1)
            y1 = self.clf_net(x1)
            x2 = self.graph_net(x2)

            pred_y1 = torch.argmax(y1, dim=1)
            pred_agreement = self.agreement_net(x1, x2)
            return pred_agreement, pred_y1

        elif self.train_mode == 'f_uu':
            x1 = input_x[0]
            x2 = input_x[1]
            x1 = self.graph_net(x1)
            x2 = self.graph_net(x2)

            pred_y1 = torch.argmax(self.clf_net(x1), dim=1)
            pred_y2 = torch.argmax(self.clf_net(x2), dim=1)
            pred_agreement = self.agreement_net(x1, x2)
            return pred_agreement, pred_y1, pred_y2

        if self.test_mode == True:
            x1 = input_x
            x1 = self.graph_net(x1)
            y_pred = self.clf_net(x1)
            return y_pred