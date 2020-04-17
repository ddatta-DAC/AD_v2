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

class gam_net(nn.Module):

    def __init__(
            self,
            input_dimension,
            encoder_dimensions,
    ):
        super(gam_net, self).__init__()
        self.setup_Net(
            input_dimension,
            encoder_dimensions
        )
        return

    def setup_Net(
            self,
            input_dimension,
            encoder_dimensions
    ):
        print(' Graph Agreement Module ')
        num_encoder_layers = len(encoder_dimensions) + 1
        self.num_encoder_layers = num_encoder_layers
        self.encoder_dimensions = encoder_dimensions
        self.encoder_layers = nn.ModuleList()

        if type(input_dimension) == list:
            num_domains = input_dimension[0]
            node_emb_dim = input_dimension[1]
            encoder_inp_dim = num_domains * node_emb_dim
        else:
            encoder_inp_dim = input_dimension

        inp_dim = encoder_inp_dim
        for i in range(self.num_encoder_layers):
            op_dim = encoder_dimensions[i-1]
            self.encoder_layers.append(nn.Linear(inp_dim, op_dim))
            inp_dim = op_dim

        # Encoder
        # 3 layer MLP
        # self.encoder = [None] * num_encoder_layers

        print('Encoder Layer :: \n', self.encoder_layers)
        # Aggregator
        # Just d = (ei -ej)^2

        # Predictor
        # 1 layer MLP
        # output should be a value
        self.predictor_layer = nn.Linear(encoder_dimensions[-1], 1)
        print('Predictor Layer ::', self.predictor_layer)
        return

    def forward(
            self,
            x1,
            x2
    ):
        if len(x1.shape) > 2:
            x1 = x1.view(-1, x1.shape[-2] * x1.shape[-1])
        if len(x2.shape) > 2:
            x2 = x2.view(-1, x2.shape[-2] * x2.shape[-1])


        e_1 = x1
        e_2 = x2

        for i in range(self.num_encoder_layers):
            e_1 = self.encoder_layers[i](e_1)
            e_2 = self.encoder_layers[i](e_2)
            e_1 = torch.tanh(e_1)
            e_2 = torch.tanh(e_2)

        # Aggregator
        d = e_1 - e_2
        d = d ** 2

        # Predictor
        res_pred = self.predictor_layer(d)
        # This should be fed to a loss function
        # torch.nn.BCELoss ; preferably torch.nn.BCEWithLogitsLoss
        # that should have inputs res_pred , agreement_indicator
        return res_pred


def gam_loss( y_pred, y_true ):
    loss_func = nn.BCEWithLogitsLoss()
    loss = loss_func(y_pred, y_true)
    return loss

# -------------------------------------------------- #

def test():
    x1 = np.random.random([10,6])
    x2 = np.random.random([10,6])
    x1 = torch.FloatTensor(x1)
    x2 = torch.FloatTensor(x2)
    net = gam_net(
        6, [6,5,4]
    )
    print(net.encoder[0].weight[0].detach().numpy())
    y = torch.FloatTensor(np.random.randint(0,1,size=[10,1]))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    optimizer.zero_grad()
    res = net.forward(x1,x2)
    loss = gam_loss(res, y)
    loss.backward()
    optimizer.step()
    print(net.encoder[0].weight[0].detach().numpy())



# ------------------------------------------ #
