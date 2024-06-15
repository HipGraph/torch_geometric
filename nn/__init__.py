import torch
import torch.nn.functional as F

import model.mytorch.util as pyt_util
from .. import util as pyg_util
from .conv.message_passing import MessagePassing


class gcLinear(MessagePassing):

    debug = 0

    def __init__(self, in_size, out_size, bias=True, order=1):
        super(gcLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn((out_size, in_size)))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_size))
        else:
            self.register_parameter("bias", None)
        self.in_size = in_size
        self.out_size = out_size
        self.order = order

    def forward(self, x, edge_index, edge_weight=None, frmt="?"):
        if self.debug:
            print("*** gcLinear Forward ***")
            print("x =", x.shape)
            print("edge_index =", edge_index.shape)
        V = x.shape[-2]
        if frmt == "?":
            if edge_index.shape[-2] == V and edge_index.shape[-1] == V:
                frmt = "adj"
            elif edge_index.shape[-2] == 2:
                frmt = "coo"
            else:
                raise ValueError("Cannot determine format of edge_index with shape=%s" % (edge_index.shape))
        if self.order & 1:
            x = F.linear(x, self.weight) if self.bias is None else F.linear(x, self.weight) + self.bias
        if frmt == "coo":
            x, edge_index = pyt_util.align((x, edge_index), -1)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            x = torch.squeeze(x, 0)
        elif frmt == "adj":
            x = self.propagate_adj(x, edge_index)
        if self.order & 2:
            x = F.linear(x, self.weight) if self.bias is None else F.linear(x, self.weight) + self.bias
        if self.debug:
            print("*** gcLinear Output ***")
            print("x =", x.shape)
        return x

    def propagate_adj(self, x, adj):
        return torch.einsum("nm,...mf->...nf", adj, x)

    def message(self, x_j, edge_weight):
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
        edge_weight = torch.unsqueeze(edge_weight, -1)
        if self.debug:
            print("x_j =", x_j.shape, "=")
            if self.debug > 1:
                print(x_j)
            print("edge_weight =", edge_weight.shape, "=")
            if self.debug > 1:
                print(edge_weight)
        return edge_weight * x_j


class gcSoftmax(torch.nn.Module):

    debug = 0

    def __init__(self):
        super(gcSoftmax, self).__init__()

    def forward(self, edge_index, edge_weight=None, frmt="adj"):
        if frmt == "adj":
            if edge_index.shape[-2] != edge_index.shape[-1]:
                raise ValueError(
                    "Format given as \"%s\" but final 2 dimensions do not match with shape=%s" % (
                        frmt, edge_index.shape
                    )
                )
            edge_index = F.softmax(edge_index, -1)
        elif frmt == "coo":
            if edge_index.shape[-2] != 2:
                raise ValueError(
                    "Format given as \"%s\" but size of second-to-last dimension != 2 with shape=%s" % (
                        frmt, edge_index.shape
                    )
                )
            edge_weight = pyg_util.normalize_edge_weight(edge_index, torch.exp(edge_weight), 1, self.debug)
        else:
            raise ValueError("Unknown format \"%s\"" % (frmt))
        return edge_index, edge_weight
