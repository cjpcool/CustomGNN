from utils import accuracy, load_data, pathsGen, consis_loss, load_large_dataset
import torch
import torch.nn.functional as F
from models import PathWeightModel
import argparse
import numpy as np
import torch.optim as optim
import time

parser = argparse.ArgumentParser()

# globla settings
parser.add_argument('--only_test', type=bool, default=False)
parser.add_argument('--para_name', type=str, default='cora')
parser.add_argument('--pw_adj_name', type=str, default='cora')

parser.add_argument('--dataset', type=str, default='cora') #CoraFull CoauthorCS CoauthorPhysics AminerCS AmazonComputers AmazonPhoto
parser.add_argument('--use_label_rate', type=bool, default=False)
parser.add_argument('--num_example_per_class', type=float, default=2)
parser.add_argument('--path', type=str, default='./data/')
parser.add_argument('--cuda',type=bool, default=True, help='ables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--cuda_device', type=int, default=0, help='cuda device')
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
parser.add_argument('--batch_size', type=int, default=300)
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
parser.add_argument('--order', type=int, default=8, help='multi-hop gnn order') # cora 8 10,  citeseer 4 5

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


optimizer = optim.Adam(model.parameters(),
                       lr=args.learning_rate, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
print('complete init model')

device = next(model.parameters()).device
print('device =', device)

iter = pathsGen(node_num, args.batch_size, graph, args.path_length, args.window_size)

# plot path weight values

def triple_loss(features,pw_adj, logs,samp_neg, samp_pos, margin=args.margin):
    global model
    label_num = int(args.pesudo_ratio * features.shape[0])

    embeddings = model.encode(features*1.0)
    idx = torch.tensor(np.random.randint(0, features.shape[0], label_num), dtype=torch.long, device=device)

    ps = [torch.exp(p)[idx] for p in logs]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)
    out = avg_p.argmax(dim=-1).view(-1,1)
    indicator_adj = (out == out.t())

    negs = torch.where((~indicator_adj).triu(1))
    poss = torch.where(indicator_adj.triu(1))

    samp_neg = samp_neg if samp_neg <= negs[0].shape[0] else negs[0].shape[0]
    samp_pos = samp_pos if samp_pos <= poss[0].shape[0] else poss[0].shape[0]
    samp_neg = np.random.randint(0,negs[0].shape[0], samp_neg)
    samp_pos = np.random.randint(0, poss[0].shape[0], samp_pos)

    loss_neg = torch.mean(F.pairwise_distance(embeddings[idx[negs[0][samp_neg]]], embeddings[idx[negs[1][samp_neg]]]))
    loss_pos = torch.mean(F.relu(margin - F.pairwise_distance(embeddings[idx[poss[0][samp_pos]]], embeddings[idx[poss[1][samp_pos]]])))

    loss = 0.
    loss = loss_pos + loss_neg
    # pw_emd = torch.matmul(pw_adj, embeddings)
    pw_emd = torch.matmul(pw_adj, embeddings*1.0)
    # pw_emd = torch.cat((gnn_emd, pw_emd), dim=1)
    loss_pos_lstm = torch.mean(F.pairwise_distance(pw_emd[idx[poss[0][samp_pos]]], pw_emd[idx[poss[1][samp_pos]]]))
    loss_neg_lstm = torch.mean(
        F.relu(margin - F.pairwise_distance(pw_emd[idx[negs[0][samp_neg]]], pw_emd[idx[negs[1][samp_neg]]])))

    loss_lstm = loss_pos_lstm + loss_neg_lstm
    all_loss = loss * args.lam_tri + loss_lstm * args.lam_tri_lstm

    return all_loss

loss_val_list = []
loss_train_list = []
acc_val_list = []
acc_train_list = []
acc_best = 0.
loss_best = 100.
loss_mn = np.inf
acc_mx = 0.0
bad_counter = 1
best_epoch = 0

def train():
    t_total = time.time()
    global iter, best_epoch, acc_best, loss_best, bad_counter, loss_mn, acc_mx
    model.train()
    for epoch in range(args.epochs):
        t = time.time()
        X_list = []
        for _ in range(args.K):
            try:
                data = next(iter)
            except StopIteration:
                iter = pathsGen(node_num, args.batch_size, graph, args.path_length, args.window_size)
                data = next(iter)
            X_list.append(data)

        # foward
        out_list=[]

        loss_train = 0.
        pw_adj = 0
        for k in range(args.K):
            out, pw = model(features, adj, *(X_list[k]))
            # embeddings.append(pw_embedding)
            out_list.append(out)
            loss_train += F.nll_loss(out_list[k][idx_train], labels[idx_train])
            pw_adj += pw

        # normalize
        rowsum = pw_adj.sum(1)
        r_inv = torch.pow(rowsum, -1).view(-1)
        r_inv[torch.isinf(r_inv)] = 0.
        pw_adj.mul_(r_inv.t())

        loss_train = loss_train / args.K
        #pw_adj /= args.K

        cons_loss = 0.
        if args.use_consis:
            cons_loss = consis_loss(out_list, args.T) * args.lam_u
            loss_train += cons_loss


        tri_loss = 0.
        if args.use_triple:
            tri_loss += triple_loss(features, pw_adj, out_list, args.samp_neg, args.samp_pos, args.margin)
            loss_train += tri_loss


        # bp
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # eval
        model.eval()
        with torch.no_grad():
            # batch = next(pathsGen(node_num, args.test_batch_size, graph, args.path_length, args.window_size))
            batch = data
            output, _ = model(features, adj, *batch, pw_adj=pw_adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_val = accuracy(output[idx_val], labels[idx_val])
            acc_train = accuracy(output[idx_train], labels[idx_train])

            loss_val_list.append(loss_val.item())
            loss_train_list.append(loss_train.item())
            acc_val_list.append(acc_val.item())
            acc_train_list.append(acc_train.item())

            spend = time.time() - t
            print('Epoch:{:04d}'.format(epoch + 1),
                  'tri:{:4f}'.format(tri_loss),
                  'consis:{:4f}'.format(cons_loss),
                  'loss_train:{:.4f}'.format(loss_train.item()),
                  'acc_train:{:.4f}'.format(acc_train.item()),
                  'loss_val:{:.4f}'.format(loss_val.item()),
                  'acc_val:{:.4f}'.format(acc_val.item()),
                  'acc_best:{:.4f}'.format(acc_best),
                  'loss_best:{:.4f}'.format(loss_best),
                  'bad_count:{:03d}'.format(bad_counter),
                  'time:{:.4f}s'.format(spend),
                  'remain_time:{:.4f}h'.format((args.epochs-epoch)*spend/3600),
                  ),

        if loss_val_list[-1] <= loss_mn or acc_val_list[-1] >= acc_mx:  # or epoch < 400:
            if acc_val_list[-1] >= acc_best: #and loss_val_list[-1] <= loss_best:  #
                loss_best = loss_val_list[-1]
                acc_best = acc_val_list[-1]
                best_epoch = epoch
                if epoch >= 0:
                    torch.save(pw_adj, dataset+'.pth')
                    torch.save(model.state_dict(), dataset + '.pkl')

            loss_mn = np.min((loss_val_list[-1], loss_mn))
            acc_mx = np.max((acc_val_list[-1], acc_mx))
            bad_counter = 1
        else:
            bad_counter += 1

        if bad_counter % args.schedule_patience == 0 and scheduler.get_lr()[-1] >= 1e-4:
            scheduler.step()
            print('schedule learning rate to',scheduler.get_lr()[-1])

        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)

            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
            break

        model.train()

def test(pw_adj_name):
    model.eval()
    with torch.no_grad():
        X = next(pathsGen(node_num, 0, graph, args.path_length, args.window_size))
        pw_adj = torch.load(pw_adj_name+'.pth').to(device)
        output, _ = model(features, adj, *X, pw_adj=pw_adj)

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    print("Test set results:",
          "loss_test= {:.4f}".format(loss_test.data.item()),
          "loss_train= {:.4f}".format(loss_train.data.item()),
          "acc_test= {:.4f}".format(acc_test.item()),
          "acc_train= {:.4f}".format(acc_train.item()))
    return loss_test.item(), loss_val.item(), loss_train.item(), acc_test.item(), acc_val.item(), acc_train.item()


if not args.only_test:
    train()
    para_name = args.dataset
    pw_adj_name = args.dataset
else:
    para_name = args.para_name
    pw_adj_name = args.pw_adj_name

## test
# Restore best model
print('Loading {}'.format(para_name))
model.load_state_dict(torch.load(para_name + '.pkl'))
model.eval()
test(pw_adj_name)
