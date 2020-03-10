import torch
import torch.nn as nn
from random import shuffle
from torch.nn.parameter import Parameter
from collections import OrderedDict
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
            emb_dim,
            domain_dims,
            num_neg_samples
    ):

        self.num_domains = len(domain_dims)
        self.domain_dims = domain_dims
        self.num_neg_samples = num_neg_samples
        self.dict_Entity_Embed = OrderedDict({})

        self.Emb = nn.Embedding(
                num_embeddings = domain_dims,
                embedding_dim = emb_dim
            )

        return

    # ---------------
    # Define network structure
    # ---------------
    def forward(
            self,
            input_x_pos,
            input_x_neg=None
    ):

        # ------------- Core function ------------------- #
        # x :: id of entity node
        # All the nodes have a contiguous id
        # ----------------------------------------------- #
        def process(
                centre_node,            # [batch, ]
                context_nodes,          # [batch, context_size]
                is_Negative = False
        ):

            x_t = self.Emb(centre_node)
            x_c = self.Emb(context_nodes)

            # For each c i.e. context
            # log (sigmoid ( x_t . x_c )


            return

        # ------------- main function ------------------- #
        if input_x_neg is None:
            return process(input_x_pos)
            # Mean so that gradients can be calculated
        else:
            pos = torch.log(process(input_x_pos))
            list_neg_x = torch.chunk(
                input_x_neg,
                self.num_neg_samples,
                dim=1
            )

            list_neg_x = [torch.squeeze(_, dim=1) for _ in list_neg_x]
            ns = []

            for _neg_x in list_neg_x:
                res = process(_neg_x, is_Negative=True)
                ns.append(res)

            ns = torch.stack(ns, dim=1)
            neg = torch.mean(ns)

            # -----------------
            # This is the objective function
            # -----------------
            res = pos + neg
            # -----------------
            # This is the objective function
            # -----------------
            # maximize P, means minimize -P
            return res

    # net.list_W_m[i].weight.detach().numpy()

class model:
    def __init__(self):
        return

    @staticmethod
    def custom_loss(y_pred, y_true=None):

        return y_pred

    def build(
            self,
            emb_dim,
            domain_dims,
            num_neg_samples = 10,
            LR = 0.005,
            num_epochs=10,
            batch_size=128,
            log_interval=100
    ):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.emb_dims = emb_dim
        domain_dims_vals = list(domain_dims.values())
        self.log_interval = log_interval
        self.net = Net()
        self.net.setup_Net(
            emb_dim,
            domain_dims_vals,
            num_neg_samples
        )
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=LR
        )
        self.criterion = model.custom_loss
        print(self.net)
        return

    def train_model(
            self,
            train_x
    ):
        bs = self.batch_size
        num_batches = train_x.shape[0] // bs + 1

        for epoch in range(self.num_epochs):
            # Shuffle
            ind_list = list(range(train_x.shape[0]))
            shuffle(ind_list)
            _train_x = train_x[ind_list, :]

            for batch_idx in range(num_batches):
                _x_pos = _train_x[batch_idx * bs:(batch_idx + 1) * bs]

                # --------
                # feed tensor
                # --------
                _x_pos = torch.LongTensor(_x_pos)

                self.optimizer.zero_grad()
                output = self.net(_x_pos)

                loss = self.criterion(
                    output,
                    None
                )
                loss.backward()
                self.optimizer.step()
                # ----- #
                if batch_idx % self.log_interval == 0:
                    print('Train ::  Epoch: {}, '
                          'Batch {}, Loss {:4f}'.format(epoch, batch_idx, loss)
                    )


# --------------------------------------------- #

obj = model()
obj.build(
    emb_dim  = 12,
    domain_dims ={'PortOFLading': 10,
                  'Carrier': 25}
)