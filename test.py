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
print('w_list:',len(w_list))
for i in w_list:
    print(i)
print('a_list',len(a_list))
for j in a_list:
    print(j)
