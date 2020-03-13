import torch
import torch.nn as nn
from random import shuffle
import numpy as np
import sys
sys.path.append('./..')
sys.path.append('./../..')
# -------------------------------- #

try :
    from src.utils import plotter
except:
    from utils import plotter

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


# ========================================================= #
# Dpmain dims should be a dictionary { 1 : val1,  2 : val2 ,.. }
# Here 1,2,3 ... are domain ids  - when they are sorted lexicographically
# ========================================================= #
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        return

    def setup_Net(
            self,
            num_entities,
            emb_dim,
            context_size,
            num_neg_samples
    ):
        self.context_size = context_size
        self.num_entities = num_entities
        self.emb_dim = emb_dim
        self.num_neg_samples = num_neg_samples

        self.Emb = nn.Embedding(
            num_embeddings=self.num_entities,
            embedding_dim=self.emb_dim
        )

        self._LogSigmoid = nn.LogSigmoid()
        self._cos = nn.CosineSimilarity(dim=-1)

        return

    # ---------------
    # Define network structure
    # ---------------
    def forward(
            self,
            input_x
    ):

        x_context = None
        x_neg_context = None
        train_mode = False
        if len(input_x) == 1:
            train_mode = False
            x_target = input_x[0]
        else:
            train_mode = True
            x_target = input_x[0]  # [ batch , 1]
            x_context = input_x[1]  # [ batch, m ]
            x_neg_context = input_x[2]  # [ batch, n, m ]

        # ------------- Core function ------------------- #
        # x :: id of entity node
        # All the nodes have a contiguous id
        # ----------------------------------------------- #
        def process(
                target,  # [batch, ]
                context,  # [batch, context_size]
                is_Negative = False
        ):

            # x_t shape [ batch, emb_dim ]
            x_t = self.Emb(target)
            # x_C should have shape [ batch, context_Size, emb_dim ]
            x_c = self.Emb(context)
            # For each c i.e. context
            # log (sigmoid ( x_t . x_c )

            if is_Negative:
                x_t1 = x_t.repeat(
                    1,
                    self.num_neg_samples * self.context_size
                ).view(
                    [-1, self.num_neg_samples, self.context_size, self.emb_dim]
                )
                p = self._cos(-x_t1, x_c)

            else:
                x_t1 = x_t.repeat(
                    1, self.context_size
                ).view([-1, self.context_size, self.emb_dim])
                p = self._cos(x_t1, x_c)

            q = self._LogSigmoid(p)
            return q

        # ------------- main function ------------------- #
        if train_mode == False:
            return self.Emb(x_target)

        else:
            # Mean so that gradients can be calculated

            pos_1 = process(
                x_target,
                x_context,
                is_Negative=False
            )

            neg_1 = process(
                x_target,
                x_neg_context,
                is_Negative=True
            )
            n2 = torch.mean(
                neg_1,
                dim=1
            )

            res = pos_1 + n2
            res = torch.sum(
                res,dim=1
            )
            res = -res
            # maximize P, means minimize -P
            return res


class model:
    def __init__(self):
        return

    @staticmethod
    def custom_loss(y_pred, y_true=None):
        return torch.mean(y_pred)

    def build(
            self,
            emb_dim,
            num_entities,
            context_size=4,
            num_neg_samples=10,
            LR=0.005,
            num_epochs=10,
            batch_size=128,
            log_interval=100
    ):
        self.context_size = context_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.num_entities = num_entities
        self.log_interval = log_interval
        self.num_neg_samples = num_neg_samples
        self.net = Net()

        self.net.setup_Net(
            num_entities=self.num_entities,
            emb_dim=self.emb_dim,
            context_size=self.context_size,
            num_neg_samples=self.num_neg_samples
        )

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=LR
        )
        self.criterion = model.custom_loss
        print(self.net)
        return

    # ===========================
    # Train the model
    # x_t : [ batch, ]
    # x_c : [ batch, context_size ]
    # x_c_neg : [ batch , num_neg_samples , context_size]
    # ===========================
    def train_model(
            self,
            x_target,  # [batch, ]
            x_context,
            x_neg_context
    ):
        bs = self.batch_size
        num_batches = x_target.shape[0] // bs + 1
        self.optimizer.zero_grad()
        record_loss = []
        for epoch in range(self.num_epochs):
            # Shuffle
            num_data_pts = x_target.shape[0]
            ind_list = list(range(num_data_pts))
            shuffle(ind_list)
            _x_t = x_target[ind_list]
            _x_c = x_context[ind_list, :]
            _x_nc = x_neg_context[ind_list, :, :]

            for batch_idx in range(num_batches):
                _x_t_b = _x_t[batch_idx * bs:(batch_idx + 1) * bs]
                _x_c_b = _x_c[batch_idx * bs:(batch_idx + 1) * bs]
                _x_nc_b = _x_nc[batch_idx * bs:(batch_idx + 1) * bs]
                # --------
                # feed tensor
                # --------
                _x_t_b = torch.LongTensor(_x_t_b)
                _x_c_b = torch.LongTensor(_x_c_b)
                _x_nc_b = torch.LongTensor(_x_nc_b)

                self.optimizer.zero_grad()
                output = self.net((_x_t_b, _x_c_b, _x_nc_b))

                loss = self.criterion(
                    output,
                    None
                )

                loss.backward()
                self.optimizer.step()
                record_loss.append(float(loss))

                # ----- #
                if batch_idx % self.log_interval == 0:
                    msg = 'Train ::  Epoch: {} Batch {}, Loss {:4f}'.format(epoch, batch_idx, loss)
                    print(msg)
        return record_loss

# --------------------------------------------- #



