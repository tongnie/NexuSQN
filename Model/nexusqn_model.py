import torch
from torch import nn
from tsl.utils.parser_utils import ArgParser
from torch.nn import functional as F
from tsl.utils.parser_utils import str_to_bool
from einops import rearrange
from einops.layers.torch import Rearrange
from tsl.nn import utils
from tsl.nn.models import BaseModel
from mlp import MultiLayerPerceptron
from embedding import PositionalEncoder


class NexuSQNModel(BaseModel):
    """
    NexuSQN: A MLP-Mixer-Based Model for Time Series Forecasting.

    Args:
        input_size (int): Size of the input.
        input_window_size (int): Size of the input window.
        input_embedding_dim (int): Size of the input projection.
        output_size (int): Size of the output.
        horizon (int): Forecasting steps.
        node_dim (int): Size of node embedding.
        exog_size (int): Size of the optional exogenous variables.
        num_layer (int): Number of dense layers in the TimeMixer.
        st_embd (int): Whether to use a spatiotemporal embedding.
    """

    def __init__(self,
                 input_size,
                 input_window_size,
                 input_embedding_dim,
                 output_size,
                 node_dim,
                 horizon,
                 n_nodes,
                 exog_size,
                 st_embd=True,
                 num_layer=2,
                 activation='gelu'):
        super(NexuSQNModel, self).__init__()

        self.input_window_size = input_window_size
        self.activation = activation
        self.st_embd = st_embd

        input_size += exog_size
        self.input_encoder = nn.Linear(input_size * input_window_size, input_embedding_dim)
        hidden_size = input_embedding_dim + node_dim

        # Spatiotemporal embeddings
        self.emb = nn.Parameter(
            torch.empty(n_nodes, node_dim))
        nn.init.xavier_uniform_(self.emb)

        self.u_enc = PositionalEncoder(in_channels=exog_size,
                                       out_channels=node_dim,
                                       n_layers=2,
                                       steps=input_window_size,
                                       n_nodes=n_nodes)

        # TimeMixer blocks
        self.TimeMixer = nn.Sequential(
            *[MultiLayerPerceptron(hidden_size, hidden_size) for _ in range(num_layer)])

        # SpaceMixer blocks
        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU())
        self.dropout1 = nn.Dropout(dropout=0)
        self.dropout2 = nn.Dropout(dropout=0)

        # Readout blocks
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            utils.get_layer_activation(self.activation)())

        self.readout = nn.Sequential(
            nn.Linear(hidden_size, horizon * output_size),
            Rearrange('b n (h f) -> b h n f', h=horizon, f=output_size))


    def forward(self, x, edge_index=None, edge_weight=None, u=None, node_index=None, **kwargs):
        """"""
        # x: [batches steps nodes features]
        x = utils.maybe_cat_exog(x, u)

        # flat time dimension
        x = rearrange(x, 'b s n f -> b n (s f)')
        x = self.input_encoder(x)  # [b n c]

        # TimeMixer with spatial context
        # add encoding
        if self.st_embd:
            q = self.u_enc(u, node_index=node_index, node_emb=self.emb)
        else:
            q = self.u_enc(u, node_index=node_index)
        x = torch.cat([x, q], dim=-1)  # [b n c]
        x = self.TimeMixer(x) + x

        # SpaceMixer, single layer implementation
        # softmax kernel method
        e = torch.softmax(self.emb, dim=-1)  # [n c]
        et = torch.softmax(self.emb.T, dim=-1)  # [c n]

        # x: (batch_size, ..., length, model_dim)
        residual = x
        # message passing
        x = et@x  # [c n] * [n c] -> [c c]
        x = e@x  # [n c] * [c c] -> [n c]
        x = self.linear(x)  # [b n c]
        x = F.gelu(x)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.feed_forward(x)  # (batch_size, ..., length, model_dim)
        x = self.dropout2(x)
        x = residual + x

        # MultiStep Readout
        x = self.decoder(x) + x
        x = self.readout(x)

        return x


    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=64, tunable=True, options=[16, 32, 64, 128, 256])
        parser.opt_list('--input-window-size', type=int, default=12, tunable=False)
        parser.opt_list('--input-embedding-dim', type=int, default=12, tunable=True)
        parser.opt_list('--node-dim', type=int, default=32, tunable=True)
        parser.opt_list('--st-embd', type=str_to_bool, nargs='?', const=False, default=True)
        parser.opt_list('--dropout', type=float, default=0., tunable=True, options=[0., 0.2, 0.3])
        parser.opt_list('--activation', type=str, default='silu', tunable=False, options=['relu', 'elu', 'silu'])
        return parser
