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


model = GAT(nfeat=1433,
                nhid=8,
                nclass=7,
                dropout=0.6,
                nheads=8,
                alpha=0.2)
model.load_state_dict(torch.load('299.pkl'))
w_list,a_list=model.show()

adj, features, labels, idx_train, idx_val, idx_test = load_data()
features = features.cuda()
adj = adj.cuda()
features, adj, labels = Variable(features), Variable(adj), Variable(labels)

def test_forward(input, adj,w,a):
        h = torch.mm(input, w)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * 8)
        leakyrelu = nn.LeakyReLU(0.2)
        e = leakyrelu(torch.matmul(a_input, a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        return attention

for k in range(8):
    temp_attention = test_forward(features,adj,w_list[k],a_list[k])
    list = temp_attention.numpy().tolist()
    print('temp_attention_',temp_attention.size())
    filename='./attention_'+str(k)+'.txt'
    with open(filename, 'w') as file_object:
        for line in list:
            for item in line:
                file_object.write(str(item) + ',')
            file_object.write('\n')
