import numpy as np
import torch
import scipy.sparse as sp

idx_features_labels = np.genfromtxt("{}{}.content".format('./data/cora/', 'cora'), dtype=np.dtype(str))
features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
features = torch.FloatTensor(np.array(features.todense()))

print(features.size())
list = features.numpy().tolist()
with open('./node_feature_01.txt', 'w') as file_object:
    for line in list:
        for item in line:
            file_object.write(str(item) + ',')
        file_object.write('\n')