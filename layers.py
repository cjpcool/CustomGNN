import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PathWeightLayer(nn.Module):
    def __init__(self, embedding_dim, lstm_hiddeen_unit=128, alpha=0.2, bias=1.):
        super(PathWeightLayer, self).__init__()
        self.lstm_hiddeen_unit = lstm_hiddeen_unit
        self.bidirectional = False
        self.bias = bias
        self.num_layers = 1
        ## you can choose a sequence module, LSTM or GRU, here,
        ## Correspondingly, please revise the forward method.
        self.lstm_layer = torch.nn.LSTM(embedding_dim, lstm_hiddeen_unit, batch_first=True,
                                        bidirectional=self.bidirectional,bias=True, num_layers=self.num_layers)
        # self.gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=lstm_hiddeen_unit, batch_first=True, bidirectional=self.bidirectional)
        self.mean_pooling = torch.nn.AvgPool1d(lstm_hiddeen_unit)
        # self.max_pooling = torch.nn.MaxPool1d(lstm_hiddeen_unit)
        # self.batch_norm = torch.nn.BatchNorm1d(lstm_hiddeen_unit)
        self.leakyrelu = nn.LeakyReLU(alpha)
        # self.bias = nn.Parameter(torch.FloatTensor(lstm_hiddeen_unit))
        self.bias=bias
        # self.reset_parameters()

    def init_hidden(self,batch_size):
        w = next(self.parameters())
        num_directions = 2 if self.bidirectional else 1
        return (w.new_zeros((num_directions*self.num_layers, batch_size, self.lstm_hiddeen_unit)).detach(),
                w.new_zeros((num_directions*self.num_layers, batch_size, self.lstm_hiddeen_unit)).detach())

    def reset_parameters(self):
        for i in range(self.num_layers):
            nn.init.orthogonal_(self.lstm_layer.all_weights[i][0])
            nn.init.orthogonal_(self.lstm_layer.all_weights[i][1])
            nn.init.zeros_(self.lstm_layer.all_weights[i][2])
            nn.init.zeros_(self.lstm_layer.all_weights[i][3])
            # nn.init.ones_(self.bias)

    def forward(self, sub_paths_emd, sub_paths_length):
        batch_size = len(sub_paths_length)
        hidden = self.init_hidden(batch_size)
        packed_path = pack_padded_sequence(sub_paths_emd, sub_paths_length, enforce_sorted=False,batch_first=True)
        _ , (lstm_out,_) = self.lstm_layer(packed_path,hidden)
        # _, lstm_out = self.gru(packed_path, hidden[0])
        # lstm_out , (_,_)= self.lstm_layer(packed_path, hidden)
        # lstm_out, _ = pad_packed_sequence(lstm_out, padding_value=0.0)

        # path_weight = F.sigmoid(lstm_out)
        path_weight = lstm_out + self.bias
        # path_weight = torch.sigmoid(lstm_out)
        path_weight = F.relu(path_weight)
        path_weight = self.mean_pooling(path_weight)[0]
        # path_weight = torch.mean(path_weight, dim=0)
        path_weight = path_weight.view(-1)
        return path_weight


class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        # nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)
            # nn.init.xavier_uniform_(self.bias.data, gain=1.414)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn=False):
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


