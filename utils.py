import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp

import torch
import _pickle as pkl
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import random

# from torch_geometric.datasets import CitationFull
# from torch_geometric.datasets import Coauthor
# from torch_geometric.datasets import AMiner
# from torch_geometric.datasets import Amazon

"""
copy from: https://github.com/shchur/gnn-benchmark/blob/master/gnnbench/data/make_dataset.py
@article{shchur2018pitfalls,
  title={Pitfalls of Graph Neural Network Evaluation},
  author={Shchur, Oleksandr and Mumme, Maximilian and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan},
  journal={Relational Representation Learning Workshop, NeurIPS 2018},
  year={2018}
}
"""
def get_train_val_test_split(random_state, labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])



def load_large_dataset(root='./data/', name='CoraFull', seed=42):

    print('loading data {} from {}'.format(name, root))

    if name == 'CoraFull':
        dataset = load_CoraFull()
    elif name == 'CoauthorCS':
        dataset = load_Coauthor(name='CS')
    elif name == 'CoauthorPhysics':
        dataset = load_Coauthor(name='Physics')
    elif name == 'AminerCS':##
        dataset = load_AMiner()
    elif name == 'AmazonComputers':
        dataset = load_Amazon(name='Computers')
    elif name == 'AmazonPhoto':
        dataset = load_Amazon(name='Photo')
    else:
        print('error dataset name input')

    print("dataset graph number={}".format(len(dataset)))
    for i in range(len(dataset)):
        print('{}th graph info:'.format(i),dataset[i])

    edge_index = dataset.data.edge_index
    features = dataset.data.x
    labels = dataset.data.y
    labels_onehot = encode_onehot(labels.numpy())
    print(labels_onehot.shape)
    num_samples, num_classes  = labels_onehot.shape
    if name == 'CoraFull': # ignore the class whose node number less than 50
        deleted_class=[]
        deleted_nodes=[]
        for class_index in [68,69]:
            nodes=[]
            for sample_index in range(num_samples):
                if labels_onehot[sample_index, class_index] > 0.0:
                    nodes.append(sample_index)
            if len(nodes) < 50:
                print('ignore class index={}, ignore node num={}'.format(class_index, len(nodes)))
                deleted_class.append(class_index)
                deleted_nodes.extend(nodes)

        features = np.delete(features, deleted_nodes, axis=0)
        labels = np.delete(labels, deleted_nodes, axis=0)
        adj = sp.coo_matrix((torch.ones_like(edge_index[0]), edge_index), shape=(num_samples, num_samples))

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize_adj(adj)
        adj = np.array(adj.todense())
        adj = np.delete(adj,deleted_nodes,axis=0)
        adj = np.delete(adj, deleted_nodes, axis=1)
        labels_onehot = encode_onehot(labels.numpy())
        graph = nx.to_dict_of_lists(nx.from_numpy_matrix(adj))
        # norm
        features = features / features.sum(1, keepdim=True).clamp(min=1)
        adj = torch.FloatTensor(adj)
    else:
        num_nodes = labels.shape[0]

        adj = sp.coo_matrix((torch.ones_like(edge_index[0]), edge_index), shape=(num_nodes,num_nodes))
        graph = nx.to_dict_of_lists(nx.from_scipy_sparse_matrix(adj))
        # norm
        features = features / features.sum(1, keepdim=True).clamp(min=1)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize_adj(adj)
        adj = torch.FloatTensor(np.array(adj.todense()))

    random_state = np.random.RandomState(seed=seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state,labels_onehot, 20, 30)

    features = torch.FloatTensor(features)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return graph, labels, adj, features, idx_test, idx_test, idx_train


def load_CoraFull():
    return CitationFull(root='./data', name="cora")

def load_Coauthor(name):
    # name = 'CS' or 'Physics'
    return Coauthor(root='./data', name=name)

def load_AMiner():
    return AMiner(root='./data')

def load_Amazon(name):
    # name = 'Computers' or 'Photo'
    return Amazon(root='./data', name=name)


def consis_loss(logps, temp):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)
    # p2 = torch.exp(logp2)

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return loss


def subPath(ind1, ind2, path, ws, padding_idx=0.):
    s = []
    dif = abs(ind1 - ind2)
    s_length = dif+1
    if ind1 > ind2:
        for i in range(ind2, ind1 + 1)[::-1]:
            s.append(int(path[i]))
        if dif < ws:
            for j in range(ws-dif):
                s.append(padding_idx)
        return s, s_length
    if ind1 < ind2:
        for i in range(ind1, ind2+1):
            s.append(int(path[i]))
        if dif < ws:
            for j in range(ws-dif):
                s.append(padding_idx)
        return s, s_length
    if ind1 == ind2:
        for i in range(ws+1):
            s.append(padding_idx)
        s[0] = int(path[ind1])
        return s, s_length


def pathsGen(node_num, batch_size, graph, path_length, window_size):
    ind = np.random.permutation(node_num)
    i = 0
    while i < ind.shape[0]:
        g = []
        sub_paths = []
        sub_paths_length = []
        # ensure j < ind.shape[0]
        q = i + batch_size - ind.shape[0]
        if q > 0:
            i -= q
            j = ind.shape[0]
        else:
            j = i + batch_size

        for k in ind[i: j]:
            if len(graph[k]) == 0:
                continue
            path = [k]  #
            for _ in range(path_length):
                # can not have cycle
                next_node = random.choice(graph[path[-1]])
                path.append(next_node)  #length = path_length
            for l in range(len(path)):
                # for m in range(max((l-window_size),0), min(l + window_size + 1, len(path))):
                for m in range(l+1, min(l + window_size + 1, len(path))):
                    if path[l] == path[m]:
                        continue
                    g.append([path[l], path[m]])
                    s, s_length = subPath(l, m, path, window_size,
                                          padding_idx=0.)  #
                    sub_paths.append(s)
                    sub_paths_length.append(s_length)

        yield \
            torch.tensor(g, dtype=torch.long),\
            torch.tensor(sub_paths, dtype=torch.long),\
            torch.tensor(sub_paths_length, dtype=torch.long)
        i = j


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(dataset, path):
    """Load data."""
    print('data loading.....')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize_features(features)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # norm
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)

    # D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    # D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    # D1 = sp.diags(D1[:, 0], format='csr')
    # D2 = sp.diags(D2[0, :], format='csr')
    # adj = adj.dot(D1)
    # adj = D2.dot(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_train = range(len(y))  # train_set include labels
    # idx_val = range(len(y), len(y) + 500)
    idx_val = test_idx_range.tolist()
    idx_test = test_idx_range.tolist()

    # print(adj.shape)  # (2708,2708)
    # print(features.shape)  # (2708,1433)
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(np.argmax(labels, -1))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('complete data loading')
    return graph, labels, adj, features, idx_val, idx_test, idx_train, y


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def plot(x1,y1,y2, label1, label2, title, save_path):
    # ploting
    print('--------------plot {}-----------------'.format(title))
    plt.plot(x1, y1, '-', label=label1)
    plt.plot(x1, y2, '-', label=label2)
    plt.legend(loc='upper left')
    plt.title(title)
    # plt.xlabel('epochs')
    plt.savefig(save_path)
    plt.close()
    print('--------------Done-----------------')


