import torch
import torch.nn as nn
from torch.nn import functional as F
from random import shuffle
import numpy as np
import sys
from torch.nn import Parameter
from torch import tensor

sys.path.append('./..')
sys.path.append('./../..')
import torch.nn.functional as F
from torch import FloatTensor as FT
from torch import LongTensor as LT

try:
    from src.Classifiers.MLP import MLP
except:
    from src.Classifiers.MLP import MLP

# -------------------------------- #


# ---------------------
# Following the paper :
# 3 layers of encoder
# d : Euclidean
# final layer : MLP
# ---------------------
class gam_net_v1(nn.Module):

    def __init__(
            self,
            input_dimension,
            encoder_dimensions,
            activation = 'relu'
    ):
        super(gam_net_v1, self).__init__()
        self.setup_Net(
            input_dimension,
            encoder_dimensions,
            activation
        )
        return

    def setup_Net(
            self,
            input_dimension,
            encoder_dimensions,
            activation
           
    ):
        print(' Graph Agreement Module ')
        self.encoder = MLP(
            input_dimension,
            encoder_dimensions,
            activation=activation,
            output_layer = False,
            output_activation = False
        )

        print('Encoder Layer :: \n', self.encoder)

        # Predictor ::
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

        e_1 = self.encoder(x1)
        e_2 = self.encoder(x2)

        # Aggregator
        # d = (ei -ej)^2
        d = (e_1 - e_2)**2

        # Predictor
        res_pred = self.predictor_layer(d)
        res_pred  = F.sigmoid(res_pred)
        # This should be fed to a loss function
        # that should have inputs ( predicted agreement, agreement_indicator )
        return res_pred

    
    
    
class agreement_net_v2(nn.Module):

    def __init__(
            self,
            input_dimension,
            encoder_dimensions,
            activation = 'relu'
    ):
        super(agreement_net_v2, self).__init__()
        self.setup_Net(
            input_dimension,
            encoder_dimensions,
            activation
        )
        return

    def setup_Net(
            self,
            input_dimension,
            encoder_dimensions,
            activation
           
    ):
        print(' Graph Agreement Module ')
        self.encoder = MLP(
            input_dimension,
            encoder_dimensions,
            activation=activation,
            output_layer = False,
            output_activation = False
        )

        print('Encoder Layer :: \n', self.encoder)

        # Predictor ::
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

        e_1 = self.encoder(x1)
        e_2 = self.encoder(x2)

        # Aggregator
        # d = (ei -ej)^2
   
        # Predictor
        res_pred  = F.sigmoid(F.cosine_similarity(e_1, e_2, dim=1))
        # This should be fed to a loss function
        # that should have inputs ( predicted agreement, agreement_indicator )
        return res_pred

    
    

def gam_loss( y_pred, y_true ):
    return  F.binary_cross_entropy(y_pred, y_true)
# -------------------------------------------------- #

def test():
    x1 = np.random.random([10,6])
    x2 = np.random.random([10,6])
    x1 = FT(x1)
    x2 = FT(x2)
    net = gam_net_v1(
        6, [6,5,4]
    )
    print(net.encoder.mlp[0].weight)
    y = torch.FloatTensor(np.random.randint(0,1,size=[10,1]))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    optimizer.zero_grad()
    res = net.forward(x1,x2)
    loss = gam_loss(res, y)
    loss.backward()
    optimizer.step()
    print(net.encoder.mlp[0].weight)



# ------------------------------------------ #
