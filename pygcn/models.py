import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
from pygcn.layers import GraphConvolution, MLPLayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False,order=1):
        super(GCN, self).__init__()
        self.order = order
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nfeat, nclass)
        self.gc2 = GraphConvolution(nhid, nclass)


        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def forward(self, x, adj):
        if self.order == 0:
            if self.use_bn:
                x = self.bn1(x)
            x = F.dropout(x, self.input_droprate, training=self.training)
            x = F.relu(self.gc4(x, adj))
            x = F.log_softmax(x)
            return x

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        
        x = F.relu(self.gc1(x, adj))

        for _ in range(self.order-1):
            if self.use_bn:
                x = self.bn2(x)
            x = F.dropout(x, self.input_droprate, training=self.training)
            x = F.relu(self.gc3(x, adj))

        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.gc2(x, adj)
        x = F.log_softmax(x)
        return x

    def get_embeddings(self, x, adj):
        if self.order == 0:
            if self.use_bn:
                x = self.bn1(x)
            x = F.dropout(x, self.input_droprate, training=self.training)
            return x

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)

        x = F.relu(self.gc1(x, adj))

        for _ in range(self.order - 1):
            if self.use_bn:
                x = self.bn2(x)
            x = F.dropout(x, self.input_droprate, training=self.training)
            x = F.relu(self.gc3(x, adj))

        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn =False):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        
    def forward(self, x):
         
        if self.use_bn: 
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        return x

