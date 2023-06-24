import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, getMADGap
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer')

parser.add_argument('--path', type=str, default='./data/')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
# parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nhid', type=int, default=128, help='Number of hidden units.')    # F'
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=250, help='Patience')

parser.add_argument('--order', type=int, default=1, help='order')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


graph, labels, adj, features, idx_val, idx_test, idx_train, y = load_data(args.dataset, args.path+args.dataset)
# Train model
def Train(order):
    args.order = order
    global adj, features, labels, idx_train, idx_val, idx_test

    model = GCN(nfeat=features.shape[1],
                nhid=args.nhid,
                nclass=int(labels.max()) + 1,
                input_droprate=args.dropout,
                hidden_droprate=args.dropout,
                order=args.order
                )
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()



        model.eval()
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_test], labels[idx_test])
        acc_val = accuracy(output[idx_test], labels[idx_test])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))
        return acc_val.data.item()



    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    best = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        acc_values.append(train(epoch))

        torch.save(model.state_dict(), 'gcn.pkl')
        if acc_values[-1] > best:
            best = acc_values[-1]
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), 'gcn_best.pkl')
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('gcn_best.pkl'))

    def compute_test(model):
        model.eval()
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        embeddings = model.get_embeddings(features, adj)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item(), embeddings
    acc, embeddings = compute_test(model)
    return acc, embeddings

# Testing
if __name__ == '__main__':
     for order in range(0,17):
         acc, embeddings = Train(order)
         madgap = getMADGap(embeddings, adj)
         with open('gcn_orders_madgap.csv', 'a+') as f:
             f.writelines(str(order)+'\t'+str(madgap)+'\n')














