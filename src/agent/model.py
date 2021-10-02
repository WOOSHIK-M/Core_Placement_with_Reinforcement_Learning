import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class GraphConv(nn.Module):
    def __init__(self, n_features: int, dropout: float = 0.5):
        super(GraphConv, self).__init__()

        self.gc1 = GraphConvolutionLayer(n_features, 32)
        self.gc2 = GraphConvolutionLayer(32, 16)
        self.gc3 = GraphConvolutionLayer(16, 1)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(F.relu(self.gc1(x, adj)), self.dropout, training=self.training)
        x = F.dropout(F.relu(self.gc2(x, adj)), self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        return x.squeeze()


class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features: int, out_features: int, bias=True, init='xavier'):
        super(GraphConvolutionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        if init == 'uniform':
            self.reset_parameters_uniform()
        elif init == 'xavier':
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            self.reset_parameters_kaiming()
        elif init == 'orthogonal':
            self.reset_parameters_orthogonal()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_orthogonal(self):
        nn.init.orthogonal_(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
