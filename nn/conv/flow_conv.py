import torch
from torch import Tensor
import torch_geometric
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor, Optional
from torch_geometric.utils.num_nodes import maybe_num_nodes

from .message_passing import MessagePassing

import Utility as util
from Models.torch import util as pyt_util


class FlowConv(MessagePassing):

    debug = 0

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        add_self_loops: bool = False, 
        normalize: int = 0,
        node_layer: str = "Linear", 
        node_act: str = "Identity", 
        edge_layer: str = "Linear", 
        edge_act: str = "Identity", 
        bias: bool = True, 
        **kwargs, 
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        if 1:
            print(
                in_channels, 
                out_channels,
                add_self_loops, 
                normalize,
                node_layer, 
                node_act, 
                edge_layer, 
                edge_act, 
                bias, 
                kwargs, 
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        if node_layer == "RNN":
            self.node_nn = torch.nn.RNN(in_channels, out_channels, batch_first=True, bias=(not bias))
        elif node_layer == "GRU":
            self.node_nn = torch.nn.GRU(in_channels, out_channels, batch_first=True, bias=(not bias))
        elif node_layer == "LSTM":
            self.node_nn = torch.nn.LSTM(in_channels, out_channels, batch_first=True, bias=(not bias))
        elif node_layer == "Linear":
#            self.layer = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
            self.node_nn = torch.nn.Linear(in_channels, out_channels, bias=(not bias))
        else:
            raise NotImplementedError(node_layer)

        if edge_act == "Identity":
            self.edge_act = torch.nn.Identity()
        elif edge_act == "Sigmoid":
            self.edge_act = torch.nn.Sigmoid()
        elif edge_act == "Tanh":
            self.edge_act = torch.nn.Tanh()
        elif edge_act == "ReLU":
            self.edge_act = torch.nn.ReLU()
        else:
            raise NotImplementedError(edge_act)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def get_sink_sums(self, edge_index, edge_weight, normalize):
        source_index = torch.squeeze(
            torch.index_select(edge_index, -2, torch.tensor(0, device=edge_weight.device)), -2
        )
        sink_index = torch.squeeze(
            torch.index_select(edge_index, -2, torch.tensor(1, device=edge_weight.device)), -2
        )
        if normalize == 1: # edge_weight.shape=(?, |E|)
            if self.debug > 0:
                print("sink_index =", sink_index.shape, "=")
                if self.debug > 1:
                    print(sink_index)
            if sink_index.shape > edge_weight.shape:
                edge_weight = edge_weight.expand(sink_index.size())
                if self.debug:
                    print("edge_weight reshaped =", edge_weight.shape, "=")
                if self.debug > 1:
                    print(edge_weight)
            sink_sums = scatter_sum(edge_weight, sink_index, dim=-1)
            if self.debug > 0:
                print("sink_sums 1. =", sink_sums.shape, "=")
                if self.debug > 1:
                    print(sink_sums)
            sink_sums = torch.take_along_dim(sink_sums, sink_index, -1)
            if self.debug > 0:
                print("sink_sums 2. =", sink_sums.shape, "=")
                if self.debug > 1:
                    print(sink_sums)
            # Check for and fix unsafe division - Summation s=0 when w(i,j)=0 for all i
            #   For e(i,j), setting s=1 gives   j = w(i,j) / s * i   =>   j = 0 / 1 * i
            #   This does not transform w(i,j) but the computation is correct with i=0
            sink_sums[sink_sums == 0] = 1
            if self.debug > 0:
                print("sink_sums 3. =", sink_sums.shape, "=")
                if self.debug > 1:
                    print(sink_sums)
        elif normalize == 2: # edge_weight.shape=(?, |V|)
            if self.debug > 0:
                print("source_index =", source_index.shape, "=")
                if self.debug > 1:
                    print(source_index)
            if self.debug > 0:
                print("sink_index =", sink_index.shape, "=")
                if self.debug > 1:
                    print(sink_index)
            source_weight = torch.take_along_dim(edge_weight, source_index, -1)
            sink_weight = torch.take_along_dim(edge_weight, sink_index, -1)
            if 0:
                if sink_index.shape > edge_weight.shape:
                    edge_weight = edge_weight.expand(sink_index.size())
                    if self.debug:
                        print("edge_weight reshaped =", edge_weight.shape, "=")
                    if self.debug > 1:
                        print(edge_weight)
            # Get summation of edge weights for sink node i across all incoming nodes j \in N(i)
            #   For method=2, this is simply the feature value (streamflow) of the sink node
            sink_sums = sink_weight
            if self.debug > 0:
                print("sink_sums 1. =", sink_sums.shape, "=")
                if self.debug > 1:
                    print(sink_sums)
            # Check for and fix unsafe division - Summation s=0 when w(j,?)=0
            #   For e(i,j), setting s=1 gives   j = w(i,j) / s * i   =>   j = w(i,j) / 1 * i
            #   This does not transform w(i,j) AND the computation is incorrect!
            #   ***The best we can do is use normalization=1 since the s in undefined in this situation
            mask = sink_sums == 0
            sink_sums[mask] = self.get_sink_sums(edge_index, source_weight, 1)[mask]
        else:
            raise NotImplementedError("Unknown option for \"normalize\" %d" % (normalize))
        return sink_sums

    def normalize_edge_weight(self, edge_index, edge_weight, normalize):
        sink_sums = self.get_sink_sums(edge_index, edge_weight, normalize)
        source_index = torch.squeeze(
            torch.index_select(edge_index, -2, torch.tensor(0, device=edge_weight.device)), -2
        )
        sink_index = torch.squeeze(
            torch.index_select(edge_index, -2, torch.tensor(1, device=edge_weight.device)), -2
        )
        if normalize == 1: # edge_weight.shape=(?, |E|)
            edge_weight = edge_weight / sink_sums
        elif normalize == 2: # edge_weight.shape=(?, |V|)
            source_weight = torch.take_along_dim(edge_weight, source_index, -1)
            edge_weight = source_weight / sink_sums
        else:
            raise NotImplementedError("Unknown option for \"normalize\" %d" % (normalize))
        return edge_weight

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        """

        Arguments
        ---------
        x : FloatTensor with shape=(?, |V|, F)
        edge_index : LongTensor with shape=(?, 2, |E|)
        edge_weight : (FloatTensor, optional) with shape=(?, |E|)


        """
        if edge_weight is None:
            x, edge_index = pyt_util.align((x, edge_index), -1)
        else:
            x, edge_index, edge_weight = pyt_util.align(
                (x, edge_index, torch.unsqueeze(edge_weight, -1)), 
                -1
            )
            edge_weight = torch.squeeze(edge_weight, -1)

        if not edge_weight is None and self.normalize:
            if self.debug:
                print(util.make_msg_block("Normalizing Edge Weight"))
            if self.debug:
                print("x =", x.shape, "=")
                if self.debug > 1:
                    print(x)
            if self.debug:
                print("edge_index =", edge_index.shape, "=")
                if self.debug > 1:
                    print(edge_index)
            if self.debug:
                print("edge_weight =", edge_weight.shape, "=")
                if self.debug > 1:
                    print(edge_weight)
            edge_weight = self.normalize_edge_weight(edge_index, edge_weight, self.normalize)

        if self.add_self_loops:
            if self.debug:
                print(util.make_msg_block("Adding Self Loops"))
            if self.debug:
                print("edge_index =", edge_index.shape, "=")
                if self.debug > 1:
                    print(edge_index)
            if self.debug and not edge_weight is None:
                print("edge_weight =", edge_weight.shape, "=")
                if self.debug > 1:
                    print(edge_weight)
            n_edges = edge_index.shape[-1]
            edge_index, _ = add_remaining_self_loops(edge_index, None, 1.0, x.shape[self.node_dim])
            n_new_edges = edge_index.shape[-1] - n_edges
            if not edge_weight is None and n_new_edges > 0:
                edge_weight = torch.cat(
                    [
                        edge_weight, 
                        torch.ones(
                            edge_weight.shape[:-1] + (n_new_edges,), 
                            dtype=torch.float, 
                            device=edge_weight.device
                        )
                    ], 
                    -1
                )
            if self.debug:
                print("edge_index =", edge_index.shape, "=")
                if self.debug > 1:
                    print(edge_index)
            if self.debug and not edge_weight is None:
                print("edge_weight =", edge_weight.shape, "=")
                if self.debug > 1:
                    print(edge_weight)

        N, T, V, F = x.shape
        if isinstance(self.node_nn, (torch.nn.RNN, torch.nn.GRU)):
            x = torch.reshape(torch.transpose(x, 1, 2), (-1, T, F))
            output, h_n = self.node_nn(x)
            x = torch.transpose(torch.reshape(output, (N, V, T, self.out_channels)), 1, 2)
        elif isinstance(self.node_nn, torch.nn.LSTM):
            x = torch.reshape(torch.transpose(x, 1, 2), (-1, T, F))
            output, (h_n, c_n) = self.node_nn(x)
            x = torch.transpose(torch.reshape(output, (N, V, T, self.out_channels)), 1, 2)
        else:
            x = self.node_nn(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        """
        Arguments
        ---------
        x_j : FloatTensor with shape=(?, |E|, F)
        edge_weight : (FloatTensor, optional) with shape=(?, |E|)

        Returns
        -------
        x : FloatTensor with shape=(?, |E|, F)

        """
        if edge_weight is None:
            return x_j
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
#        edge_weight, x_j = pyt_util.align((edge_weight, x_j), -1)
        edge_weight = torch.unsqueeze(edge_weight, -1)
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
        return self.edge_act(edge_weight) * x_j

    def reset_parameters(self):
        self.node_nn.reset_parameters()
        zeros(self.bias)
#        self.bias = torch.nn.init.zeros_(self.bias)
