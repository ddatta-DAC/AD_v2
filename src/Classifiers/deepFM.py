import pandas as pd
import numpy as np
import sys

sys.path.append('./..')
sys.path.append('./../..')
from torch import nn
import torch
from torch import tensor

import multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.data
import yaml
import pickle
import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import LongTensor as LT
import torch.nn.functional as F
try:
    from .MLP import MLP
except:
    from MLP import MLP


# ------------------------------------------------ #

class clf_dfm(nn.Module):

    def __init__(
            self,
            wide_inp_01_dim, # the size of the 1-0 vector
            num_domains,  # number of fields
            entity_emb_dimensions,  # emb dimension of each entity
            FM_inp_dim=None,
            dnn_layer_dimensions=None,  # FNN for concatenated embedding
            pretrained_node_embeddings=None,  # type : FloatTensor,
            tune_entity_emb=False,
            num_entities=None  # total number of entities (reqd if no pretrained emb)
    ):

        super(clf_dfm, self).__init__()

        self.wide_inp_dim = wide_inp_01_dim
        self.wide_Linear = nn.Linear(wide_inp_01_dim, 1)

        if pretrained_node_embeddings is not None:
            self.embedding = torch.nn.Embedding.from_pretrained(
                pretrained_node_embeddings,
                freeze=tune_entity_emb
            )
        else:
            self.embedding = torch.nn.Embedding(
                num_entities,
                entity_emb_dimensions
            )

        self.entity_emb_dims = entity_emb_dimensions
        self.num_domains = num_domains
        self.concat_emb_dim = self.num_domains * self.entity_emb_dims
        self.dnn_num_layers = len(dnn_layer_dimensions) + 1  # Number of fully connected layers

        inp_dim = self.entity_emb_dims * self.num_domains  # Concatenation
        self.dnn_fc = MLP(
            inp_dim,
            dnn_layer_dimensions
        )
        if FM_inp_dim is None :
            self.transform_emb = False
        else:
            self.transform_emb = True
            self.num_domains = num_domains
            self.xform_FM_1 = nn.ModuleList(
                [
                    nn.Linear(self.entity_emb_dims, FM_inp_dim)
                    for _ in range(self.num_domains)
                ]
            )



        return

    def forward(self, input_x):
        """
        input x should be of shape [Batch, onehot_size + num_domains]
        """

        x_wide = input_x[:, :self.wide_inp_dim]
        x_deep = input_x[:, self.wide_inp_dim:]

        # Pass input through embedding look up
        print(x_deep.shape)
        x = self.embedding(x_deep)
        print(x.shape)

        # ----- DNN ------- #
        x_dnn = x.view(-1, self.concat_emb_dim)
        print(x_dnn.shape)
        x_dnn = self.dnn_fc(x_dnn)

        # ----- FM --------- #

        fm_input = x

        if self.transform_emb:
            # transform inner products with a MLP
            fm_input = torch.chunk(
                fm_input,
                self.num_domains,
                dim=1
            )

            x_fm_input = []
            for i in range(self.num_domains):
                x_fm_input.append(
                    self.xform_FM_1[i](fm_input[i]).squeeze(1)
                )
            x_fm_input = torch.stack(
                x_fm_input,
                dim=1
            )
        else:
            x_fm_input = fm_input

        s1 = torch.sum(x_fm_input, dim=1, keepdim=True)
        square_of_sum = s1 ** 2
        sum_of_square = torch.sum(x_fm_input ** 2, dim=1, keepdim=True)

        cross_term = square_of_sum - sum_of_square
        InnerProduct = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        x_op = self.wide_Linear(x_wide.float()) + x_dnn + InnerProduct
        x_op = torch.nn.functional.sigmoid(x_op)

        return x_op






def test():
    model = clf_dfm(
        wide_inp_01_dim = 7,
        num_domains=3,
        entity_emb_dimensions=4,
        FM_inp_dim=8,
        dnn_layer_dimensions=[5, 6],
        pretrained_node_embeddings=torch.FloatTensor(np.random.random([15, 4]))
    )

    x0 = np.random.randint(0, 2, size = [16, 7])
    x1 = np.random.randint(0, 2, size= [16, 3])
    x2 = np.hstack([x0,x1])
    x = LT(x2)

    print(' Input :: ', x.shape)
    y_pred = model(x)
    y_true = FT(np.random.randint(0, 2, size =[16, 1]))
    criterion =  nn.BCELoss()
    loss = criterion( y_pred, y_true)

    # loss.backward()

test()


