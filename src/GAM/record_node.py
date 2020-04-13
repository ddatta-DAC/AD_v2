#!/usr/bin/env python
#-*- coding: utf-8 -*-

# ---------------
# Author : Debanjan Datta
# Email : ddatta@vt.edu
# ---------------

import torch
import torch.nn as nn
from random import shuffle
import numpy as np
import sys
from torch.nn import Parameter
from torch import tensor

sys.path.append('./..')
sys.path.append('./../..')
import torch.nn.functional as F

# -------------------------------- #


# ---------------------------------- #
# Following GraphSage
# Concatenate the node embeddings
# And pass them through activation (ReLu)
# ---------------------------------- #
class graph_net_v1(nn.Module):
    def __init__(
            self,
            emb_dimension,
            num_domains,
            output_dimensions,
            pretrained_node_embeddings
    ):
        super(graph_net_v1, self).__init__()
        self.emb_dimension = emb_dimension
        self.num_domains = num_domains
        self.setup_Net(
            pretrained_node_embeddings,
            output_dimensions
        )

        return

    def setup_Net(
            self,
            pretrained_node_embeddings,
            output_dimensions
    ):
        self.embedding = torch.nn.Embedding.from_pretrained(
            pretrained_node_embeddings,
            freeze=True
        )
        self.fc1 = torch.nn.Linear(
            self.num_domains * self.emb_dimension,
            output_dimensions
        )

        return

    # Input is a tensor/array shape [?, num_domains, serial_entity_id]
    # Output :
    # shape : [ ?, output_dimension ]
    def forward(self, input_x):
        x = self.embedding(input_x)
        x = x.view(-1, self.num_domains * self.emb_dimension)
        x = self.fc1(x)
        # print(' [Forward] graph net ', x.shape)
        # x = nn.LeakyReLU(x)
        return x

# ------------------------------------------------------------------------------- #










