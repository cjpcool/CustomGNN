from __future__ import division
from __future__ import print_function

import random
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import load_data, accuracy, load_large_dataset, sample_per_class, encode_onehot, load_syn_cora
from models import MLP
from layers import MLPLayer
from random import sample
import random

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='syn-cora')
parser.add_argument('--homophily_ratio_name', type=str, default='h0.20-r1')
parser.add_argument('--use_label_rate', type=bool, default=False)
parser.add_argument('--num_example_per_class', type=float, default=2)
parser.add_argument('--path', type=str, default='./data/')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode_rate', type=float, default=0.0,
                    help='Dropnode rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--order', type=int, default=4, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')

parser.add_argument('--use_embed', type=bool, default=False)# 50 135 128 256 for citeseer cora pubmed nell
parser.add_argument('--embedding_dim', type=int, default=256)# 50 135 128 256 for citeseer cora pubmed nell

parser.add_argument('--use_triple', type=bool, default=False)# 50 135 128 256 for citeseer cora pubmed nell
parser.add_argument('--samp_neg', type=int, default=10000, help='negative pairs of triple_loss.')
parser.add_argument('--samp_pos', type=int, default=10000, help='positive pairs of triple_loss.')
parser.add_argument('--lam_tri', type=int, default=1.0, help='the weight of triple loss')
parser.add_argument('--margin', type=float, default=1., help='margin between negative samples')

#dataset = 'citeseer'
#dataset = 'pubmed'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.cuda_device)
dataset = args.dataset
# np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# _, labels, A, features, idx_val, idx_test, idx_train, y = load_large_dataset(args.path, args.dataset)
if args.dataset in ['cora','citeseer', 'pubmed']:
    _, labels, A, features, idx_val, idx_test, idx_train, y = load_data(args.dataset, './data/' + args.dataset)
elif args.dataset in ['syn-cora', 'syn-product']:
    graph, labels, A, features, idx_val, idx_test, idx_train = load_syn_cora(args.dataset, args.path, args.homophily_ratio_name, args.seed)

if args.use_label_rate:
    seed = random.randint(0,200)
    print(seed)
    torch.manual_seed(seed)
    random_state = np.random.RandomState(seed=seed)
    num_example_per_class = args.num_example_per_class
    idx_train = torch.LongTensor(sample_per_class(random_state, encode_onehot(labels.numpy()), num_example_per_class,
                                 forbidden_indices=idx_test))
    
idx_unlabel = torch.range(idx_train.shape[0], labels.shape[0]-1, dtype=int)

# Model and optimizer
embedding_dim = args.embedding_dim if args.use_embed else features.shape[1]

weight = nn.Parameter(torch.FloatTensor(features.shape[1], embedding_dim))
nn.init.xavier_uniform_(weight.data, gain=1.414)
def encode(W, input):
    return F.leaky_relu(torch.mm(input, W))

model = MLP(nfeat=embedding_dim,
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            input_droprate=args.input_droprate,
            hidden_droprate=args.hidden_droprate,
            use_bn=args.use_bn)

optimizer = optim.Adam([{'params': model.parameters()},
                        {'params': weight}
                        ],
                       lr=args.lr, weight_decay=args.weight_decay)



if args.cuda:
    model.cuda()
    weight = weight.cuda()
    features = features.cuda()
    A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_unlabel = idx_unlabel.cuda()


def triple_loss(features, indicator_adj, samp_neg, samp_pos, margin=args.margin):
    device = device = next(model.parameters()).device
    embeddings = encode(weight, features)
    poss = torch.where(indicator_adj.add(torch.eye(indicator_adj.shape[0],device=device)*-1))
    negs = torch.where(indicator_adj==0)
    samp_neg = samp_neg if samp_neg <= negs[0].shape[0] else negs[0].shape[0]
    samp_pos = samp_pos if samp_pos <= poss[0].shape[0] else poss[0].shape[0]

    ind_pos = torch.tensor(sample(poss[0].tolist(), samp_pos)).to(device)
    ind_neg = torch.tensor(sample(negs[0].tolist(), samp_neg)).to(device)
    # ind_pos = torch.multinomial(torch.ones(poss[0].shape[0], dtype=torch.float), samp_pos).to(device)
    # ind_neg = torch.multinomial(torch.ones(poss[0].shape[0], dtype=torch.float), samp_neg).to(device)
    loss_pos = F.pairwise_distance(embeddings[poss[0][ind_pos]], embeddings[poss[1][ind_pos]]).mean()
    loss_neg = F.relu(margin - F.pairwise_distance(embeddings[negs[0][ind_neg]],embeddings[negs[1][ind_neg]])).mean()
    # print(loss_pos)
    # print(loss_neg)
    loss = loss_pos + loss_neg
    return loss

def propagate(feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    if args.use_embed:
        feature = encode(weight, feature)

    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        #print(y.add_(x))
        y.add_(x)
        
    return y.div_(order+1.0)

def rand_prop(features, training):
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    
    if training:
            
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)

        features = masks.cuda() * features
            
    else:
        features = features * (1. - drop_rate)

    features = propagate(features, A, args.order)    
    return features

def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss

def train(epoch):
    t = time.time()
    
    X = features
    
    model.train()
    optimizer.zero_grad()
    X_list = []
    K = args.sample
    for k in range(K):
        X_list.append(rand_prop(X, training=True))

    output_list = []
    for k in range(K):
        output_list.append(torch.log_softmax(model(X_list[k]), dim=-1))

    
    loss_train = 0.
    for k in range(K):
        loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
     
        
    loss_train = loss_train/K
    #loss_train = F.nll_loss(output_1[idx_train], labels[idx_train]) + F.nll_loss(output_1[idx_train], labels[idx_train])
    #loss_js = js_loss(output_1[idx_unlabel], output_2[idx_unlabel])
    # loss_en = entropy_loss(output_1[idx_unlabel]) + entropy_loss(output_2[idx_unlabel])

    loss_consis = consis_loss(output_list)

    loss_train = loss_train + loss_consis

    tri_loss = 0.
    if args.use_triple:
        # indicator_adj = get_indicator_adj(output_list, args.sample)
        tri_loss = triple_loss(features, None, args.samp_neg, args.samp_pos, args.margin) * args.lam_tri
        loss_train += tri_loss


    acc_train = accuracy(output_list[0][idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        X = rand_prop(X,training=False)
        output = model(X)
        output = torch.log_softmax(output, dim=-1)
        
    loss_val = F.nll_loss(output[idx_val], labels[idx_val]) 
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_tri: {:4f}'.format(tri_loss),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t), end='\t')
    return loss_val.item(), acc_val.item()


def Train():
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    # best = args.epochs + 1
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0

    for epoch in range(args.epochs):
        # if epoch < 200:
        #   l, a = train(epoch, True)
        #   loss_values.append(l)
        #   acc_values.append(a)
        #   continue

        l, a = train(epoch)
        loss_values.append(l)
        acc_values.append(a)


        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if  acc_values[-1] >= acc_best: #and:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), dataset +'grand'+'.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        print('best_acc: {:4f}'.format(acc_best), 'bad_count:{}'.format(bad_counter))

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.6f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(dataset +'grand'+'.pkl'))



def test():
    model.eval()
    X = features
    X = rand_prop(X, training=False)
    output = model(X)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss = {:.4f}".format(loss_test.item()),
          "accuracy = {:.4f}".format(acc_test.item()))
    return acc_test.item()


max = 0.
min = 100.
avg = 0.
times = 1
for d in ['h0.00-r1']:
    args.homophily_ratio_name = d
    for j in range(times):
        seed = random.randint(0, 200)
        if args.use_label_rate:
            print(seed)
            random_state = np.random.RandomState(seed=seed)
            idx_train = torch.LongTensor(
                sample_per_class(random_state, encode_onehot(labels.cpu().numpy()), args.num_example_per_class,
                                 forbidden_indices=idx_test))
        # idx_unlabel = torch.range(idx_train.shape[0], labels.shape[0] - 1, dtype=int)
        graph, labels, A, features, idx_val, idx_test, idx_train = load_syn_cora(args.dataset, args.path,
                                                                                 args.homophily_ratio_name, seed)

        # Model and optimizer
        embedding_dim = args.embedding_dim if args.use_embed else features.shape[1]

        model = MLP(nfeat=embedding_dim,
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    input_droprate=args.input_droprate,
                    hidden_droprate=args.hidden_droprate,
                    use_bn=args.use_bn)

        optimizer = optim.Adam([{'params': model.parameters()},
                                ],
                               lr=args.lr, weight_decay=args.weight_decay)

        if args.cuda:
            model.cuda()
            weight = weight.cuda()
            features = features.cuda()
            A = A.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
            idx_unlabel = idx_unlabel.cuda()

        Train()
        acc = test()
        print('acc:', acc)
        if acc > max:
            max = acc
        if acc < min:
            min = acc
        avg += acc
        with open('./' + args.dataset + '_grand_hrate{}_'.format(d) + '.csv', 'a+') as f:
            f.writelines(str(max) + '\t' + str(min) + '\t' + str(acc) + '\n')
    avg /= times
    with open('./' + args.dataset + '_grand_hrate{}_'.format(d) + '.csv', 'a+') as f:
        f.writelines(str(max) + '\t' + str(min) + '\t' + str(avg) + '\n')



