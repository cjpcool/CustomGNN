from utils import accuracy, load_data, pathsGen, consis_loss, load_large_dataset
import torch
import torch.nn.functional as F
from models import PathWeightModel
import argparse
import numpy as np
import torch.optim as optim
import time
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
parser = argparse.ArgumentParser()

# globla settings
parser.add_argument('--only_test', type=bool, default=False)
parser.add_argument('--para_name', type=str, default='citeseer_example')#cora_849_1
parser.add_argument('--pw_adj_name', type=str, default='citeseer')

parser.add_argument('--dataset', type=str, default='citeseer') #CoraFull CoauthorCS CoauthorPhysics AminerCS AmazonComputers AmazonPhoto
parser.add_argument('--use_label_rate', type=bool, default=False)
parser.add_argument('--num_example_per_class', type=float, default=2)
parser.add_argument('--path', type=str, default='./data/')
parser.add_argument('--cuda',type=bool, default=False, help='ables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--cuda_device', type=int, default=1, help='cuda device')
parser.add_argument('--patience', type=int, default=300, help='patience')
parser.add_argument('--schedule_patience', type=int, default=50, help='schedule_patience')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lam_pw_emd', type=float, default=1.0, help='Number of epochs to train.')   #citeseer 5 10; cora 0.1 0.5;

# triplet loss settings
parser.add_argument('--use_triple', type=bool, default=True)# 50 135 128 256 for citeseer cora pubmed nell
parser.add_argument('--samp_neg', type=int, default=5000, help='negative pairs of triple_loss.')
parser.add_argument('--samp_pos', type=int, default=15000, help='positive pairs of triple_loss.')
parser.add_argument('--lam_tri', type=int, default=.1, help='the weight of triple loss') # 3=85.1, 74.3,
parser.add_argument('--lam_tri_lstm', type=int, default=1., help='the weight of triple loss')
parser.add_argument('--margin', type=float, default=.1, help='margin between negative samples')
parser.add_argument('--pesudo_ratio', type=float, default=1., help='number of pesudo labels for triple loss')

# lr
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')

# batch size
parser.add_argument('--batch_size', type=int, default=1000)
# consistency loss settings
parser.add_argument('--use_consis', type=bool, default=True)
parser.add_argument('--T', type=int, default=0.5, help='temperature')
parser.add_argument('--K', type=int, default=4, help='number of batch per epoch')
parser.add_argument('--lam_u', type=int, default=1.,
                    help='tradeoff between sup loss and unsup loss, loss=sup_loss + lam_u * unsup_loss')

#Path Attention module settings
parser.add_argument('--embedding_dim', type=int, default=512)
parser.add_argument('--window_size', type=int, default=6)  # cora3,10; citeseer 456
parser.add_argument('--path_length', type=int, default=10)
parser.add_argument('--lstm_hidden_units', type=int, default=128)

# MLP settings
parser.add_argument('--nhid', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--dropout_pw', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout_adj', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout_enc', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout_input', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout_hidden', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--order', type=int, default=6, help='multi-hop gnn order') # cora 8 10,  citeseer 4 5



args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
torch.cuda.set_device(args.cuda_device)
dataset = args.dataset
# np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    # device = 'cuda:{}'.format(args.cuda_device)

# load data
if args.dataset in ['CoraFull', 'CoauthorCS', 'AmazonComputers', 'AmazonPhoto']:
    graph, labels, adj, features, idx_val, idx_test, idx_train = load_large_dataset(root=args.path, name=args.dataset)
elif args.dataset in ['cora', 'citeseer', 'pubmed']:
    graph, labels, adj, features, idx_val, idx_test, idx_train, _ = load_data(args.dataset, args.path+args.dataset)

node_num = len(graph)
nclass = int(labels.max()) + 1
embedding_dim =args.embedding_dim

print('init model ...')
model = PathWeightModel(
    features.shape[1],
    args.nhid,
    nclass,
    args.lstm_hidden_units,
    node_num,
    args.order,
    embedding_dim,
    args.lam_pw_emd,
    args.alpha,
    args.dropout_pw,
    args.dropout_adj,
    args.dropout_enc,
    args.dropout_input,
    args.dropout_hidden
)



if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
print('complete init model')


para_name = args.para_name
pw_adj_name = args.pw_adj_name

print('Loading {}'.format(para_name))
model.load_state_dict(torch.load(para_name + '.pkl'))
model.eval()
device = next(model.parameters()).device

iter = pathsGen(node_num, args.batch_size, graph, args.path_length, args.window_size)
data = next(iter)
def path_length():
    model.eval()
    with torch.no_grad():
        pairs, sub_paths, sub_path_length = data
        # pw_adj = 0.
        # for k in range(args.K):
        #     _, pw = model(features, adj, *(X))
        #     pw_adj += pw
        # pw_adj /= args.K
        nodes_embedding = model.encode(features)
        sub_paths_emd = nodes_embedding[sub_paths]
        pw_ij_batch = model.PWLayer(sub_paths_emd, sub_path_length)

        count = np.array([0 for _ in range(args.window_size+1)], dtype=float)
        sum = np.array([0 for _ in range(args.window_size+1)], dtype=float)
        for i, l in enumerate(sub_path_length):
            sum[l-1] += pw_ij_batch[i].item()
            count[l-1] += 1
        sum = sum / count
        x = [i for i in range(1,args.window_size+1)]
        y = sum[1:]
        # x = sub_path_length
        # y = pw_ij_batch.cpu()
        # sns.set_context('paper')
        df = pd.DataFrame(data={'Weight':y, 'Length':x})
        fig = sns.lmplot(x='Length',y='Weight',data=df)
        fig.fig.set_size_inches(2.3,1.7)
        fig.set(ylim=(0.,.9))
        plt.savefig('./imgs/{}_weight.pdf'.format(args.dataset),bbox_inches='tight',dpi=1024)


def test_diversity():
    model.eval()
    with torch.no_grad():
        pairs, sub_paths, sub_path_length = data
        # pw_adj = 0.
        # for k in range(args.K):
        #     _, pw = model(features, adj, *(X))
        #     pw_adj += pw
        # pw_adj /= args.K
        nodes_embedding = model.encode(features)
        sub_paths_emd = nodes_embedding[sub_paths]
        pw_ij_batch = model.PWLayer(sub_paths_emd, sub_path_length)

        count = np.array([0 for _ in range(nclass)], dtype=float)
        sum = np.array([0 for _ in range(nclass)], dtype=float)
        for i in range(sub_paths.shape[0]):
            if sub_path_length[i] != 8:
                continue
            t = np.zeros((20,), dtype=int)
            for j in range(args.window_size):
                id = sub_paths[i][j]
                if id == 0:
                    break
                t[labels[id]] = 1
            d = t.sum()
            count[d] += 1
            sum[d] += pw_ij_batch[i]
        sum = sum / count
        print(count)
        print(sum)
        x = [i for i in range(nclass)]
        y = sum
        # x = sub_path_length
        # y = pw_ij_batch.cpu()

        df = pd.DataFrame(data={'Weight':y, 'Diversity':x})
        fig = sns.lmplot(x='Diversity',y='Weight',data=df)
        fig.fig.set_size_inches(2.3,1.7)
        # fig.set(ylim=(0.,.7))
        plt.savefig('./imgs/{}_diversity.pdf'.format(args.dataset),bbox_inches='tight',dpi=1024)

# test_diversity()

def test_semantics():
    model.eval()
    with torch.no_grad():
        pairs, sub_paths, sub_path_length = data
        # pw_adj = 0.
        # for k in range(args.K):
        #     _, pw = model(features, adj, *(X))
        #     pw_adj += pw
        # pw_adj /= args.K
        nodes_embedding = model.encode(features)
        sub_paths_emd = nodes_embedding[sub_paths]
        pw_ij_batch = model.PWLayer(sub_paths_emd, sub_path_length)

        count = np.zeros((nclass,nclass), dtype=float)
        sum =  np.zeros((nclass,nclass), dtype=float)
        paper_num = np.zeros((nclass,), dtype=float)
        tag = np.zeros_like(labels.cpu())
        for i in range(sub_paths.shape[0]):
            l = sub_path_length[i]
            start, end = sub_paths[i][0], sub_paths[i][l-1]
            t_start, t_end = labels[start], labels[end]
            count[t_start][t_end] += 1
            sum[t_start][t_end] += pw_ij_batch[i]
            if tag[start] == 0:
                paper_num[t_start] += 1
                tag[start] += 1
        print(paper_num)
        print(paper_num.sum())

        # sum += sum.T
        # count += count.T
        sum = sum / count

        print(count)
        print(sum)
        # y = np.concatenate([sum, sum.mean(axis=1,keepdims=True)], axis=1)
        y = sum[::-1,:].T
        x = ['Agents','AI','DB','IR','ML','HCI']
        print(x[::-1])
        mask = np.zeros_like(sum)
        # mask[np.triu_indices_from(mask)] = True
        # mask = mask-np.eye(y.shape[0])
        # mask = np.concatenate([mask, np.zeros((mask.shape[0], 1))], axis=1)
        # print(mask.shape)
        print(y.shape)
        # sns.set_context('paper')
        plt.figure(figsize=(4.1,3.3))
        # print(mask)
        fig = sns.heatmap(data=y, yticklabels=x, xticklabels=x[::-1], cmap='YlGnBu')# YlGnBu, hot_r
        # fig.fig.set_size_inches(4.8,3.5)
        # fig.set(ylim=(0.,.7))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=14)

        plt.savefig('./imgs/{}_semantics0_30.pdf'.format(args.dataset),bbox_inches='tight',dpi=1024)

def diversity_length():
    model.eval()
    with torch.no_grad():
        pairs, sub_paths, sub_path_length = data

        nodes_embedding = model.encode(features)
        sub_paths_emd = nodes_embedding[sub_paths]
        pw_ij_batch = model.PWLayer(sub_paths_emd, sub_path_length)

        count = np.array([0 for _ in range(args.window_size + 1)], dtype=float)
        sum = np.array([0 for _ in range(args.window_size + 1)], dtype=float)
        for i, l in enumerate(sub_path_length):
            sum[l - 1] += pw_ij_batch[i].item()
            count[l - 1] += 1
        sum = sum / count
        x = [i for i in range(1, args.window_size + 1)]
        y = sum[1:]
        # x = sub_path_length
        # y = pw_ij_batch.cpu()

        df = pd.DataFrame(data={'weight': y, 'path_length': x})

        fig, axes = plt.subplots(1,2, figsize=(5, 2))

        ax0 = sns.lmplot(x='path_length', y='weight', data=df)


        pairs, sub_paths, sub_path_length = data

        nodes_embedding = model.encode(features)
        sub_paths_emd = nodes_embedding[sub_paths]
        pw_ij_batch = model.PWLayer(sub_paths_emd, sub_path_length)

        count = np.array([0 for _ in range(nclass)], dtype=float)
        sum = np.array([0 for _ in range(nclass)], dtype=float)
        for i in range(sub_paths.shape[0]):
            if sub_path_length[i] != 8:
                continue
            t = np.zeros((20,), dtype=int)
            for j in range(args.window_size):
                id = sub_paths[i][j]
                if id == 0:
                    break
                t[labels[id]] = 1
            d = t.sum()
            count[d] += 1
            sum[d] += pw_ij_batch[i]
        sum = sum / count
        print(count)
        print(sum)
        x = [i for i in range(nclass)]
        y = sum

        df = pd.DataFrame(data={'weight': y, 'diversity': x})
        ax1 = sns.lmplot(x='diversity', y='weight', data=df)
        # fig.fig.set_size_inches(2.5, 2)
        # ax1.set(ylim=(0.,.7))

        plt.savefig('./imgs/{}_weight_diversity.pdf'.format(args.dataset), bbox_inches='tight', dpi=1024)


diversity_length()