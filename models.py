import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer,MyLayer,OneLayer
import numpy as np
from utils import read_entropy_attention_list

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        #original--attention
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(nheads)]

        #entropy--attention
        attentionlist=read_entropy_attention_list()
        self.attentions = [MyLayer(nfeat, nhid, attentionlist[i] ,dropout=dropout,concat=True) for i in range(nheads)]

        # array = open('../edGNN_entropy/bin/preprocessed_data/citeseer/citeseer/citeseer_adj.txt').readlines()
        # matrix = []
        # for line in array:
        #     line = line.strip('\n').strip(',').split(',')
        #     line = [int(x) for x in line]
        #     matrix.append(line)
        # matrix = np.array(matrix)
        # adj1 = torch.FloatTensor(matrix)

        #one-attention-layer
        #self.attentions = [OneLayer(nfeat, nhid, dropout=dropout, adj=adj1,concat=True) for i in range(nheads)]

        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        #simple--gnn
        #self.oneatt=OneLayer(nfeat, nclass, dropout=dropout, adj=adj1,concat=False)

        #simple--attenetion-1
        #self.entropy_attention=MyLayer(nfeat, nclass, attentionlist[0] ,dropout=dropout,concat=False)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        #x=F.elu(self.oneatt(x,adj))

        return F.log_softmax(x, dim=1)

    def show(self):
        w_list=[]
        a_list=[]
        for i, attention in enumerate(self.attentions):
            w_list.append(attention.W)
            a_list.append(attention.a)
        return w_list,a_list

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

