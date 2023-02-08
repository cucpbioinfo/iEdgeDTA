import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
# from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

######################################################################################################################       

class GraphConv(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_input_dim=None, batchnorm=False, add_self_loops=True, bias=True):
        super(GraphConv, self).__init__(aggr='add')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.add_self_loops = add_self_loops
        
        if batchnorm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
            
        """if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation"""
            
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim, bias=True)
        else:
            self.edge_linear = None

        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x, edge_index, edge_attr):
        # Step1: Add self loop
        if self.add_self_loops==True:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # If GCN : Compute normalization --> (inverse(sqrt)(D))(A)(inverse(sqrt)(D))
        row, col = edge_index.type(torch.long)
        deg = degree(col, x.size(0), dtype=x.dtype).unsqueeze(-1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x, norm=norm)

    def  message(self, edge_attr, x_j, norm):
        # add self loop 
        # graph object : Data(x=[num_node, n_features], edge_index=[2, num_edge], edge_attr=[num_edge, e_features])
      
        message = x_j

        if self.edge_linear:
                edge_input = self.edge_linear(edge_attr.float())
                message += edge_input
            
        return norm.view(-1, 1) * message
    
    def update(self, aggr_out):
        aggr_out = self.linear(aggr_out)
        #if self.bias is not None:
        #    aggr_out += self.bias
        if self.batch_norm:
            aggr_out = self.batch_norm(aggr_out)
        """if self.activation:
            aggr_out = self.activation(aggr_out)"""
        return aggr_out
