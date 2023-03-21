import torch
import torch_geometric
import inspect
from torch_scatter import scatter_sum
from torch_geometric.typing import Adj, OptTensor, PairTensor, Optional

import Utility as util
from .nn.conv.flow_conv import FlowConv


# Setup str -> function maps for all PyTorch and PYG functions/classes needed for model definition.
#   These maps allow users to call, for example, the LSTM module using layer_fn_map["LSTM"]().
#   This allows for higher-level model definitions that are agnostic of chosen layers, activations, etc.
#   For example, we can now define a cell-agnostic RNN model by passing a layer argument (e.g. layer="LSTM")
#       that specifies the specific layer to use. Thus, this model would define a general RNN architecture,
#       such as sequence-to-sequence, with the specific layer type as a hyper-parameter.

#   init gcn layer -> class constructor function map
gcnlayer_fn_map = {}
for name, fn in inspect.getmembers(torch_geometric.nn.conv, inspect.isclass):
    if issubclass(fn, torch_geometric.nn.conv.MessagePassing) and not name in ["MessagePassing"]:
        gcnlayer_fn_map[name] = fn
for name, fn in inspect.getmembers(torch_geometric.nn.dense, inspect.isclass):
    if name.endswith("Conv"):
        gcnlayer_fn_map[name] = fn
gcnlayer_fn_map["FlowConv"] = FlowConv

#   init gcn layer -> supported feature list map
gcnlayer_supported_map = {
    "APPNP": ["SparseTensor", "edge_weight", "static"],
    "ARMAConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "CGConv": ["SparseTensor", "edge_attr", "bipartite", "static"],
    "ChebConv": ["edge_weight", "static", "lazy"],
    "DNAConv": ["SparseTensor", "edge_weight"],
    "FAConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "FlowConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "GCNConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "GCN2Conv": ["SparseTensor", "edge_weight", "static"],
    "GraphConv": ["SparseTensor", "edge_weight", "bipartite","static", "lazy"],
    "GatedGraphConv": ["SparseTensor", "edge_weight", "static"],
    "LEConv": ["SparseTensor", "edge_weight", "bipartite", "static", "lazy"],
    "LGConv": ["SparseTensor", "edge_weight", "static"],
    "ResGatedGraphConv": ["SparseTensor", "bipartite", "static", "lazy"],
    "SAGEConv": ["SparseTensor", "bipartite", "static", "lazy"],
    "SGConv": ["SparseTensor", "edge_weight", "static", "lazy"],
    "TAGConv": ["SparseTensor", "edge_weight", "static", "lazy"],
}

#   init gcn layer -> requirements dict map
gcnlayer_required_map = {
    "AGNNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "APPNP": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["?", "|E|"], 
    },
    "ARMAConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "CGConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "ChebConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "ClusterGCNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "DenseGCNConv": {
        "inputs": ["x", "adj"],
        "x.shape": ["?", "|V|", "F"],
        "adj.shape": ["?", "|V|", "|V|"],
    },
    "DenseGINConv": {
        "inputs": ["x", "adj"],
        "x.shape": ["?", "|V|", "F"],
        "adj.shape": ["?", "|V|", "|V|"],
    },
    "DenseGraphConv": {
        "inputs": ["x", "adj"],
        "x.shape": ["N", "|V|", "F"],
        "adj.shape": ["N", "|V|", "|V|"],
    },
    "DenseSAGEConv": {
        "inputs": ["x", "adj"],
        "x.shape": ["?", "|V|", "F"],
        "adj.shape": ["?", "|V|", "|V|"],
    },
    "DNAConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "DynamicEdgeConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "ECConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "EdgeConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "EGConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FAConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FastRGCNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FeaStConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FlowConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "FiLMConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["?", "|E|"], 
    },
    "GatedGraphConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GATConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GATv2Conv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GCN2Conv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GCNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "GENConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GeneralConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GINConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GINEConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GMMConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "GraphConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "GravNetConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "HGTConv": {
        "inputs": ["x_dict", "edge_index_dict"],
        "x_dict.shape": ["|V|", "F"],
        "edge_index_dict.shape": [2, "|E|"],
    },
    "HypergraphConv": {
        "inputs": ["x", "hyperedge_index"],
        "x.shape": ["|V|", "F"],
        "hyperedge_index.shape": ["|V|", "|E|"],
    },
    "LEConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "LEConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "MFConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "NNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "PANConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "PDNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "PNAConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "PointConv": {
        "inputs": ["x", "pos", "edge_index"],
        "x.shape": ["|V|", "F"],
        "pos.shape": ["|V|", 3],
        "edge_index.shape": [2, "|E|"],
    },
    "PointNetConv": {
        "inputs": ["x", "pos", "edge_index"],
        "x.shape": ["|V|", "F"],
        "pos.shape": ["|V|", 3],
        "edge_index.shape": [2, "|E|"],
    },
    "PPFConv": {
        "inputs": ["x", "pos", "normal", "edge_index"],
        "x.shape": ["|V|", "F"],
        "pos.shape": ["|V|", 3],
        "normal.shape": ["|V|", 3],
        "edge_index.shape": [2, "|E|"],
    },
    "ResGatedGraphConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "RGCNConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "SAGEConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "SGConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "SignedConv": {
        "inputs": ["x", "pos_edge_index", "neg_edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "pos_edge_index.shape": [2, "|E^(+)|"],
        "neg_edge_index.shape": [2, "|E^(-)|"],
    },
    "SplineConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
    "SuperGATConv": {
        "inputs": ["x", "edge_index", "neg_edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "neg_edge_index.shape": [2, "|E^(-)|"],
    },
    "TAGConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["?", "|V|", "F"],
        "edge_index.shape": [2, "|E|"],
        "edge_weight.shape": ["|E|"], 
    },
    "TransformerConv": {
        "inputs": ["x", "edge_index"],
        "x.shape": ["|V|", "F"],
        "edge_index.shape": [2, "|E|"],
    },
}


def add_remaining_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None, fill_value: float = 1., num_nodes: Optional[int] = None):
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted and already contains a few self-loops, only
    non-existent self-loops will be added with edge weights denoted by
    :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if edge_index.dim() > 2:
        orig_size = edge_index.size()
        n_edges = orig_size[-1]
        edge_index = edge_index.view(-1, 2, n_edges)
        edge_index_set, edge_weight_set = [], []
        for i in range(edge_index.size(0)):
            _edge_index = edge_index[i]
            _edge_weight = edge_weight
            if not edge_weight is None:
                _edge_weight = edge_weight[i]
            _edge_index, _edge_weight = torch_geometric.utils.add_remaining_self_loops(
                _edge_index, _edge_weight, fill_value, num_nodes
            )
            edge_index_set.append(_edge_index)
            edge_weight_set.append(_edge_weight)
        try:
            edge_index = torch.cat(edge_index_set)
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                raise RuntimeError(
                    (
                        "Failed to add self loops to dynamic edge_index because number "
                        "of added self loops was not uniform across instances resulting "
                        "in a ragged tensor."
                    )
                )
        edge_index = edge_index.view(orig_size[:-1] + edge_index.size()[-1:])
        if not edge_weight is None:
            edge_weight = torch.cat(edge_weight_set, 0)
            edge_weight = edge_weight.view(orig_size[:-1] + edge_weight.size()[-1:])
        return edge_index, edge_weight
    return torch_geometric.utils.add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)


def get_sink_sums(edge_index, edge_weight, normalize, debug=0):
    source_index = torch.squeeze(
        torch.index_select(edge_index, -2, torch.tensor(0, device=edge_weight.device)), -2
    )
    sink_index = torch.squeeze(
        torch.index_select(edge_index, -2, torch.tensor(1, device=edge_weight.device)), -2
    )
    if normalize == 1: # edge_weight.shape=(?, |E|)
        if debug:
            print("sink_index =", sink_index.shape, "=")
            if debug > 1:
                print(sink_index)
        if sink_index.shape > edge_weight.shape:
            edge_weight = edge_weight.expand(sink_index.size())
            if debug:
                print("edge_weight reshaped =", edge_weight.shape, "=")
            if debug > 1:
                print(edge_weight)
        sink_sums = scatter_sum(edge_weight, sink_index, dim=-1)
        if debug:
            print("sink_sums 1. =", sink_sums.shape, "=")
            if debug > 1:
                print(sink_sums)
        sink_sums = torch.take_along_dim(sink_sums, sink_index, -1)
        if debug:
            print("sink_sums 2. =", sink_sums.shape, "=")
            if debug > 1:
                print(sink_sums)
        # Check for and fix unsafe division - Summation s=0 when w(i,j)=0 for all i
        #   For e(i,j), setting s=1 gives   j = w(i,j) / s * i   =>   j = 0 / 1 * i
        #   This does not transform w(i,j) but the computation is correct with i=0
        sink_sums[sink_sums == 0] = 1
        if debug:
            print("sink_sums 3. =", sink_sums.shape, "=")
            if debug > 1:
                print(sink_sums)
    elif normalize == 2: # edge_weight.shape=(?, |V|)
        if debug:
            print("source_index =", source_index.shape, "=")
            if debug > 1:
                print(source_index)
        if debug:
            print("sink_index =", sink_index.shape, "=")
            if debug > 1:
                print(sink_index)
        source_weight = torch.take_along_dim(edge_weight, source_index, -1)
        sink_weight = torch.take_along_dim(edge_weight, sink_index, -1)
        if 0:
            if sink_index.shape > edge_weight.shape:
                edge_weight = edge_weight.expand(sink_index.size())
                if debug:
                    print("edge_weight reshaped =", edge_weight.shape, "=")
                if debug > 1:
                    print(edge_weight)
        # Get summation of edge weights for sink node i across all incoming nodes j \in N(i)
        #   For method=2, this is simply the feature value (streamflow) of the sink node
        sink_sums = sink_weight
        if debug:
            print("sink_sums 1. =", sink_sums.shape, "=")
            if debug > 1:
                print(sink_sums)
        # Check for and fix unsafe division - Summation s=0 when w(j,?)=0
        #   For e(i,j), setting s=1 gives   j = w(i,j) / s * i   =>   j = w(i,j) / 1 * i
        #   This does not transform w(i,j) AND the computation is incorrect!
        #   ***The best we can do is use normalization=1 since the s in undefined in this situation
        mask = sink_sums == 0
        sink_sums[mask] = get_sink_sums(edge_index, source_weight, 1)[mask]
    else:
        raise NotImplementedError("Unknown option for \"normalize\" %d" % (normalize))
    return sink_sums

def normalize_edge_weight(edge_index, edge_weight, normalize, debug=0):
    sink_sums = get_sink_sums(edge_index, edge_weight, normalize, debug)
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
