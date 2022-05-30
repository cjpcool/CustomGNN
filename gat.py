import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import AMiner
from torch_geometric.datasets import Amazon
import torch_geometric.transforms as T
import scipy.sparse as sp
import networkx as nx


import csv
# from utils import MADGap
from utils import load_CoraFull, load_Coauthor, load_Amazon, encode_onehot, get_train_val_test_split, normalize_adj

def load_dataset(root='./data/',name='AmazonCS', seed=42):
    print('loading data {} from {}'.format(name, root))
    if name == 'CoraFull':
        dataset = load_CoraFull()
    elif name == 'CoauthorCS':
        dataset = load_Coauthor(name='CS')
    elif name == 'AmazonComputers':
        dataset = load_Amazon(name='Computers')
    elif name == 'AmazonPhoto':
        dataset = load_Amazon(name='Photo')
    else:
        print('error dataset name input')
    edge_index = dataset.data.edge_index
    labels = dataset.data.y
    labels_onehot = encode_onehot(labels.numpy())
    print(labels_onehot.shape)
    num_samples, num_classes  = labels_onehot.shape
    if name == 'CoraFull':  # ignore the class whose node number less than 50
        deleted_class = []
        deleted_nodes = []
        for class_index in [68, 69]:
            nodes = []
            for sample_index in range(num_samples):
                if labels_onehot[sample_index, class_index] > 0.0:
                    nodes.append(sample_index)
            if len(nodes) < 50:
                print('ignore class index={}, ignore node num={}'.format(class_index, len(nodes)))
                deleted_class.append(class_index)
                deleted_nodes.extend(nodes)

        adj = sp.coo_matrix((torch.ones_like(edge_index[0]), edge_index), shape=(num_samples, num_samples))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = np.array(adj.todense())
        adj = np.delete(adj, deleted_nodes, axis=0)
        adj = np.delete(adj, deleted_nodes, axis=1)
        new_edge_index = nx.to_edgelist(nx.from_numpy_matrix(adj))
        new_edge_index = torch.LongTensor(list(map(lambda x: x[:2], new_edge_index))).t()
        new_edge_index_b = torch.vstack((new_edge_index[1],new_edge_index[0]))
        new_edge_index = torch.cat((new_edge_index, new_edge_index_b), dim=1)

        features = np.delete(dataset.data.x, deleted_nodes, axis=0)
        labels = np.delete(labels, deleted_nodes, axis=0)
        labels_onehot = encode_onehot(labels.numpy())

        dataset.data.edge_index = new_edge_index
        dataset.data.y = torch.LongTensor(labels)
        dataset.data.x = torch.FloatTensor(features)

        print(dataset.data.edge_index.shape)
        print(dataset.data.x.shape)
        print(dataset.data.y.shape)

    labels = dataset.data.y
    labels_onehot = encode_onehot(labels.numpy())
    random_state = np.random.RandomState(seed=seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state,labels_onehot, 20, 30)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_test)
    idx_test = torch.LongTensor(idx_test)
    dataset.train_mask = idx_train
    dataset.test_mask = idx_test
    return dataset, idx_train, idx_test

def get_mask(dataset, seed):
    labels = dataset.data.y
    labels_onehot = encode_onehot(labels.numpy())
    random_state = np.random.RandomState(seed=seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels_onehot, 20, 30)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_test)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_test


# 'Cora' or 'CiteSeer'
name = 'AmazonPhoto'
# 'MADGap' or 'Accuracy'
evaluate_name = 'Accuracy'

torch.manual_seed(72)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)

dataset, _, _ = load_dataset('./data/', name=name, seed=42)
dataset.transform = T.NormalizeFeatures()

def main(order):
    nhid = 8
    nhid_layers = []
    for i in range(order-1):
        nhid_layers.append(nhid)

    class GAT(nn.Module):
        def __init__(self):
            super(GAT, self).__init__()
            self.in_head = 8
            self.out_head = 1

            self.list = [dataset.num_features] + nhid_layers + [dataset.num_classes]
            self.layers = []
            self.layers.append(GATConv(self.list[0], self.list[1], heads=self.in_head, dropout=0.6))
            # self.list = [16, 8, 8, 8, 3]
            for i, _ in enumerate(self.list[1:-2]):
                self.layers.append(GATConv(nhid*self.list[i+1], self.list[i+2], heads=self.in_head, dropout=0.6))
            # 最后一层由于heads不同，单独拿出来
            self.layers.append(GATConv(nhid*self.list[-2], self.list[-1], heads=self.out_head, dropout=0.6))
            self.layers = ListModule(*self.layers)

        def forward(self, data):
            features, edges = data.x, data.edge_index
            if nhid_layers:
                for i, _ in enumerate(self.list[:-2]):
                    features = F.relu(self.layers[i](features, edges))
                    # 从第二层开始进行dropout
                    if i > 0:
                        features = F.dropout(features, p=0.6, training=self.training)
                # 最后一层
                features = self.layers[i+1](features, edges)
            else:
                features = self.layers[0](features, edges)
            output = F.log_softmax(features, dim=1)

            return output, features

    class ListModule(nn.Module):
        """
        Abstract list layer class.
        """
        def __init__(self, *args):
            """
            Module initializing.
            """
            super(ListModule, self).__init__()
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

        def __getitem__(self, idx):
            """
            Getting the indexed layer.
            """
            if idx < 0 or idx >= len(self._modules):
                raise IndexError('index {} is out of range'.format(idx))
            it = iter(self._modules.values())
            for i in range(idx):
                next(it)
            return next(it)

        def __iter__(self):
            """
            Iterating on the layers.
            """
            return iter(self._modules.values())

        def __len__(self):
            """
            Number of layers.
            """
            return len(self._modules)

    model = GAT().to(device)
    data = dataset[0].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # train
    model.train()
    acc_best = 0
    patience = 200
    bad_count = 0
    for epoch in range(10000):

        model.train()
        optimizer.zero_grad()
        output, features = model(data)
        loss = F.nll_loss(output[dataset.train_mask], data.y[dataset.train_mask])
        loss.backward()
        optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            output, features = model(data)
            pred = output.argmax(1)
            correct = float(pred[dataset.test_mask].eq(data.y[dataset.test_mask]).sum().item())
            acc = correct / dataset.test_mask.shape[0]
            if acc > acc_best:
                acc_best = acc
            else:
                bad_count += 1
            if bad_count > patience:
                print('early stop')
                break
        model.train()
    model.eval()
    output, features = model(data)
    pred = output.argmax(1)
    # print(pred)
    if evaluate_name == 'Accuracy':
        correct = float(pred[dataset.test_mask].eq(data.y[dataset.test_mask]).sum().item())
        acc = correct / dataset.test_mask.shape[0]
        return acc
    elif evaluate_name == 'MADGap':
        madgap = MADGap(dataset, features.cpu().detach().numpy())
        madgap_value = madgap.mad_gap_regularizer()
        return madgap_value
    else:
        print('evaluate_name error')

for name in ['CoraFull']:
    print('begin train',name)
    out = 0
    order=2
    max = 0
    min = 100
    seed = random.randint(0, 100)
    dataset,_,_ = load_dataset('./data/', name=name, seed=seed)
    dataset.transform = T.NormalizeFeatures()
    for _ in range(0, 50):
        seed = random.randint(0, 100)
        train_mask, test_mask = get_mask(dataset, seed)
        dataset.train_mask = train_mask
        dataset.test_mask = test_mask
        print('training gat seed={} ....'.format(seed))
        acc = main(order)
        if acc > max:
            max = acc
        if acc < min:
            min = acc
        print('seed {}, acc {}'.format(seed, acc))
        out += acc
    out = out / 50
    print('acc:', out)
    with open('./' + name + '_gat_' + evaluate_name + '.csv', 'a+') as f:
        f.writelines(str(max) + '\t' +str(min)+'\t'+ str(out) + '\n')

