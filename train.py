from __future__ import division
from __future__ import print_function

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

from utils import load_data, accuracy
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--entropy', type=int, default=0, help='If need entropy')
parser.add_argument('--gpu', type=int, default=-1, help='-1 for cpu')

args = parser.parse_args()

if args.gpu < 0:
    cuda = False
else:
    cuda = True
    torch.cuda.set_device(args.gpu)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)



# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

#
# array = open('../edGNN_entropy/bin/preprocessed_data/citeseer/citeseer/citeseer_adj.txt').readlines()
# matrix = []
# for line in array:
#     line = line.strip('\n').strip(',').split(',')
#     line = [int(x) for x in line]
#     matrix.append(line)
# matrix = np.array(matrix)
# adj=torch.FloatTensor(matrix)
#
# node_feature = []
# node_feature_file = open('../edGNN_entropy/bin/preprocessed_data/citeseer/citeseer/node_features.txt', "r").readlines()
# for line in node_feature_file:
#     vector = [float(x) for x in line.strip('\n').strip(',').split(",")]
#     node_feature.append(vector)
# features= torch.FloatTensor(np.array(node_feature))
#
# labels = []
# node_label_file = open('../edGNN_entropy/bin/preprocessed_data/citeseer/citeseer/node_labels.txt', "r").readlines()
# for line in node_label_file:
#     labels.append(int(line))
# nodN = len(labels)
# labels=torch.LongTensor(np.array(labels))
#
# #mask
# random_idx=[i for i in range(nodN)]
# random.shuffle(random_idx)
# train_idx=random_idx[nodN//5:]
# idx_test=random_idx[:nodN//5]
# idx_val = train_idx[:len(train_idx) // 5]
# idx_train = train_idx[len(train_idx) // 5:]
#
# idx_test=torch.LongTensor(idx_test)
# idx_val=torch.LongTensor(idx_val)
# idx_train=torch.LongTensor(idx_train)


print('adj:',adj.size())
print('features:',features.size())
print('labels:',labels.size())
print('train_idx:',idx_train.size())
print('idx_val:',idx_val.size())
print('idx_test:',idx_test.size())



# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if cuda:
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

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print('bad counter: ',bad_counter)
# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
os.remove('{}.pkl'.format(best_epoch))
# Testing
acc=compute_test()
with open('./cora_acc.txt', 'a+') as f:
    f.write(str(args.dropout)+' '+str(args.lr)+' : '+str(acc) + '\n')
