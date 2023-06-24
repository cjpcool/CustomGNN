import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MLP
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


    class MLP1(nn.Module):
        def __init__(self):
            nhid = 128
            super(MLP1, self).__init__()
            self.mlp = MLP(nfeat=dataset.num_features, nhid=nhid, nclass=dataset.num_classes,
                hidden_droprate=0.5, input_droprate=0.5)

        def forward(self, features):
            out = self.mlp(features)
            return F.log_softmax(out)

    model = MLP1().to(device)
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
        output = model(data.x)
        loss = F.nll_loss(output[dataset.train_mask], data.y[dataset.train_mask])
        loss.backward()
        optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            output = model(data.x)
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
    output = model(data.x)
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

for name in ['AmazonPhoto','AmazonComputers','CoauthorCS', 'CoraFull']:
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
    with open('./' + name + '_mlp_' + evaluate_name + '.csv', 'a+') as f:
        f.writelines(str(max) + '\t' +str(min)+'\t'+ str(out) + '\n')

