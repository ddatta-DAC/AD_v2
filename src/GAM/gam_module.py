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


try:
    print('Cuda available ::', torch.cuda.is_available(), 'Cde current device ::', torch.cuda.current_device(),
          torch.cuda.get_device_name(0))
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
except:
    print('No CUDA')


class gam_net(nn.Module):

    def __init__(
            self,
            node_input_dimension,
            encoder_dimensions,
    ):
        super(gam_net, self).__init__()
        self.setup_Net(
            node_input_dimension,
            encoder_dimensions
        )
        return

    def setup_Net(
            self,
            node_input_dimension,
            encoder_op_dimensions
    ):
        num_encoder_layers = len(encoder_op_dimensions)
        self.num_encoder_layers = num_encoder_layers
        self.encoder_dimensions = encoder_op_dimensions

        # Encoder
        # 3 layer MLP
        self.encoder = [None] * num_encoder_layers
        inp_dim = node_input_dimension
        print('Encoder Layer ::')
        for i in range(num_encoder_layers):
            if i == num_encoder_layers -1 :
                op_dim = 2
            else:
                op_dim = encoder_op_dimensions[i]
            self.encoder[i] = nn.Linear(inp_dim, op_dim)
            self.register_parameter('encoder_' + str(i), self.encoder[i].weight)
            print(self.encoder[i])
            inp_dim = op_dim


        # Aggregator
        # Just d = (ei -ej)^2

        # Predictor
        # 1 layer MLP
        # output should be a value
        self.predictor_layer = nn.Linear(encoder_op_dimensions[-1], 1)
        print('Predictor Layer ::', self.predictor_layer)
        return

    def forward(
            self,
            x1,
            x2
    ):

        e_1 = x1
        e_2 = x2
        for i in range(self.num_encoder_layers):
            print(i)
            print(e_1.shape, e_2.shape)
            print(self.encoder[i].weight.shape)
            e_1 = self.encoder[i](e_1)
            e_2 = self.encoder[i](e_2)
            e_1 = torch.tanh(e_1)
            e_2 = torch.tanh(e_2)

            print('-----')

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
    return  loss

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

test()