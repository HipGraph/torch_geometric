import os
import sys
import torch
import numpy as np
import torch_scatter as scat

from nn.conv.gcn_conv import GCNConv


def make_msg_block(msg, block_char="#"):
    msg_line = 3*block_char + " " + msg + " " + 3*block_char
    msg_line_len = len(msg_line)
    msg_block = "%s\n%s\n%s" % (
        msg_line_len*block_char,
        msg_line,
        msg_line_len*block_char
    )
    return msg_block


def get_test_data(shape):
    return torch.arange(np.prod(shape), dtype=torch.float).reshape(shape) + 1


def get_test_edges(version):
    if version == "static":
        return torch.tensor(
                [
                    [0,1], 
                    [1,1]
                ], 
            dtype=torch.long
        )
    elif version == "dynamic":
        return torch.tensor(
            [
                [
                    [0,1], 
                    [0,1]
                ], 
                [
                    [0,1], 
                    [1,1]
                ], 
                [
                    [0,1], 
                    [0,0]
                ], 
                [
                    [0,1], 
                    [1,0]
                ], 
            ], 
            dtype=torch.long
        )
    raise NotImplementedError


def get_scenario(scenario):
    if scenario == 0:
        print(make_msg_block("Static Features, Static Edges, Static Weights"))
        x = get_test_data((2, 1))
        edge_index = get_test_edges("static")
        edge_weight = get_test_data((2,))
    elif scenario == 1:
        print(make_msg_block("Static Features, Static Edges, Dynamic Weights"))
        x = get_test_data((2, 1))
        edge_index = get_test_edges("static")
        edge_weight = get_test_data((4, 2))
    elif scenario == 2:
        print(make_msg_block("Static Features, Dynamic Edges, Static Weights"))
        x = get_test_data((2, 1))
        edge_index = get_test_edges("dynamic")
        edge_weight = get_test_data((2,))
    elif scenario == 3:
        print(make_msg_block("Static Features, Dynamic Edges, Dynamic Weights"))
        x = get_test_data((2, 1))
        edge_index = get_test_edges("dynamic")
        edge_weight = get_test_data((4, 2))
    elif scenario == 4:
        print(make_msg_block("Dynamic Features, Static Edges, Static Weights"))
        x = get_test_data((4, 2, 1))
        edge_index = get_test_edges("static")
        edge_weight = get_test_data((2,))
    elif scenario == 5:
        print(make_msg_block("Dynamic Features, Static Edges, Dynamic Weights"))
        x = get_test_data((4, 2, 1))
        edge_index = get_test_edges("static")
        edge_weight = get_test_data((4, 2))
    elif scenario == 6:
        print(make_msg_block("Dynamic Features, Dynamic Edges, Static Weights"))
        x = get_test_data((4, 2, 1))
        edge_index = get_test_edges("dynamic")
        edge_weight = get_test_data((2,))
    elif scenario == 7:
        print(make_msg_block("Dynamic Features, Dynamic Edges, Dynamic Weights"))
        x = get_test_data((4, 2, 1))
        edge_index = get_test_edges("dynamic")
        edge_weight = get_test_data((4, 2))
    else:
        raise NotImplementedError()
    return x, edge_index, edge_weight


scenarios = map(int, sys.argv[1].split(","))
torch.manual_seed(0)
model = GCNConv(1, 1, normalize=False)
for i in scenarios:
    x, edge_index, edge_weight = get_scenario(i)
    print("x =", x.shape, "=")
    print(x)
    print("edge_index =", edge_index.shape, "=")
    print(edge_index)
    print("edge_weight =", edge_weight.shape, "=")
    print(edge_weight)
    y = model(x, edge_index, edge_weight)
    print("y =", y.shape, "=")
    print(y)
