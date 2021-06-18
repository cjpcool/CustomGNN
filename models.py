import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MLP, PathWeightLayer, GraphAttentionLayer, MLPLayer
import numpy as np


class PathWeightModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, lstm_hidden_unit, node_num, order, embedding_dim, lam_pw_emd,alpha=0.2,
                 dropout_pw=0.6, dropout_adj=0.6, dropout_enc=0.5, dropout_input=0.5,dropout_hidden=0.5):
        """for train1.py, return both pw_adj and predicted softmax"""
        super(PathWeightModel, self).__init__()
        self.dropout_pw = dropout_pw
        self.dropout_adj = dropout_adj
        self.dropout_enc = dropout_enc
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden

        self.node_num = node_num
        self.order = order
        self.lam_pw_emd = lam_pw_emd
        self.W_pw = nn.Parameter(torch.FloatTensor(nfeat, embedding_dim))
        # self.bias_pw = nn.Parameter(torch.FloatTensor(embedding_dim))
        nn.init.xavier_uniform_(self.W_pw.data, gain=1.414)
        # nn.init.zeros_(self.bias_pw)

        self.PWLayer = PathWeightLayer(embedding_dim, lstm_hidden_unit)
        self.MLP = MLP(2*embedding_dim, nhid, nclass, self.dropout_input, self.dropout_hidden)
        # self.MLP_GRAND = MLP(nfeat, 32, nclass, self.dropout_input, self.dropout_hidden)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.norm = nn.BatchNorm1d(embedding_dim)

    def get_pw_adj(self, nodes_embedding, pairs, sub_paths, sub_path_length):
        device = next(self.parameters()).device
        # device = torch.cuda.current_device()
        # compute pw_ij
        pairs = pairs.to(device)
        sub_paths = nodes_embedding[sub_paths]
        pw_ij_batch = self.PWLayer(sub_paths, sub_path_length)
        pw_adj = torch.sparse.FloatTensor(pairs.t(), pw_ij_batch,
                                          torch.Size([self.node_num, self.node_num]))

        # cont_ij = torch.ones_like(pw_ij_batch)
        # count_adj = torch.sparse.FloatTensor(pairs.t(), cont_ij,
        #                                   torch.Size([self.node_num, self.node_num]))
        #
        # pw_adj = pw_adj.mul(torch.pow(count_adj, -1)).to_dense()
        # pw_adj.add_(torch.eye(self.node_num, device=device))

        pw_adj = pw_adj.to_dense()
        eye = torch.eye(self.node_num, device=device)
        #

        pw_adj.add_(eye)

        zero_vec = -9e15 * torch.ones_like(pw_adj)
        pw_adj = torch.where(pw_adj > 0, pw_adj, zero_vec)
        pw_adj = F.softmax(pw_adj, dim=1)
        return pw_adj

    def encode(self, input):
        return F.relu(torch.mm(input, F.dropout(self.W_pw, self.dropout_enc,self.training)))

    def decode(self, input):
        return torch.mm(input, self.W.t())

    def propagation(self, features, adj, training=False, order=1):
        adj = F.dropout(adj,self.dropout_adj, training=training)
        x = features.clone().detach_()
        y = features
        for _ in range(order):
            x = torch.matmul(adj, x).detach_()
            y.add_(x)
        y = y.div_(order+1)
        return y

    def forward(self, features, adj, pairs, sub_paths, sub_path_length, pw_adj=None):
        # features = F.dropout(features, self.dropout, self.training)
        # print(nodes_embedding.sum())
        nodes_embedding = self.encode(features*1.0)
        # nodes_embedding = features*1.0
        # gnn_emd = self.encode(self.W_pw, features)
        gnn_emd = self.propagation(nodes_embedding, adj, training=self.training, order=self.order)
        # gnn_emd = self.norm(gnn_emd)
        if pw_adj is None:
            pw_adj = self.get_pw_adj(gnn_emd, pairs, sub_paths, sub_path_length)
        pw_adj1 = F.dropout(pw_adj, self.dropout_pw, training=self.training)
        pw_emd = self.lam_pw_emd * torch.matmul(pw_adj1, nodes_embedding)
        # pw_emd = self.norm(pw_emd)
        embedding_prim = torch.cat((gnn_emd, pw_emd), dim=1)
        # embedding_prim = pw_emd
        # MLP
        # embedding_prim = gnn_emd.add(pw_emd)

        out = self.MLP(embedding_prim)

        return out.log_softmax(dim=-1), pw_adj

    def getEmbeddings(self, features, adj, pairs, sub_paths, sub_path_length, pw_adj=None):
        # features = F.dropout(features, self.dropout, self.training)
        # print(nodes_embedding.sum())
        # nodes_embedding = self.encode(features)
        nodes_embedding = features*1,0
        # gnn_emd = self.encode(self.W_pw, features)
        gnn_emd = self.propagation(nodes_embedding, adj, training=self.training, order=self.order)
        if pw_adj is None:
            pw_adj = self.get_pw_adj(nodes_embedding, pairs, sub_paths, sub_path_length)
        pw_adj1 = F.dropout(pw_adj, self.dropout_pw, training=self.training)
        pw_emd = self.lam_pw_emd * torch.matmul(pw_adj1, nodes_embedding)
        embedding_prim = torch.cat((gnn_emd, pw_emd), dim=1)
        return embedding_prim


