import numpy as np
import scipy.sparse as sp
import torch
import random


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    nodN=2708
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


    #print('adj_symmetric:', adj)
    #np.savetxt('./data/cora/adj.csv',np.array(adj.todense()) , delimiter=",", fmt='%s')
    features = normalize_features(features)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))


    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])


    random_idx = [i for i in range(nodN)]
    random.shuffle(random_idx)
    train_idx = random_idx[nodN // 5:]
    test_idx = random_idx[:nodN // 5]
    val_idx = train_idx[:len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]


    train_idx=torch.LongTensor(train_idx)
    print('train_idx:',train_idx.size())
    val_idx=torch.LongTensor(val_idx)
    print('val_idx:',val_idx.size())
    test_idx = torch.LongTensor(test_idx)
    print('test_idx:',test_idx.size())


    return adj, features, labels, train_idx, val_idx, test_idx


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

def read_csv(graphfile_csv='../edGNN_entropy/bin/preprocessed_data/cora/adj.csv'):
    array = open(graphfile_csv).readlines()
    N = len(array)
    matrix = []
    for line in array:
        line = line.strip('\r\n').split(',')
        line = [float(x) for x in line]
        matrix.append(line)
    matrix = np.array(matrix)
    return torch.from_numpy(matrix).float()


def read_txt(graphfile_txt='../edGNN_entropy/bin/preprocessed_data/citeseer/citeseer/citeseer_adj.txt'):
    array = open(graphfile_txt).readlines()
    matrix = []
    for line in array:
        line = line.strip('\n').strip(',').split(',')
        line = [int(x) for x in line]
        matrix.append(line)
    adj = torch.from_numpy(np.array(matrix))
    return adj


def read_entropy_vector_list():
    print('loading entropy vectors')
    #cora
    entropy_vector=[5291.57282775566, 5930.17339392563, 35919.4807569822,
            3996.33297329960, 5985.41022812210, 45880.6612182431, 2563.09578325316, 3762.89089310395]
    entropy_vector_tensor=torch.from_numpy(np.array(entropy_vector)).float()
    motif_num=[1071, 549, 1630, 320, 380, 1489, 220, 249]

    out_list=[]
    for i in range(8):
        out_list.append(motif_num[i]*entropy_vector_tensor)

    return out_list

def read_entropy_attention_list():
    print('loading entropy as attention...')

    #citeseer
    #nodN=3312

    #cora
    nodN=2708

    #../edGNN_entropy/bin/preprocessed_data/cora/edge_entropy.txt
    #../edGNN_entropy/bin/preprocessed_data/citeseer/citeseer/citeseer_edge_entropy.txt

    #read edge entropy
    edge_entropy_file = open('../edGNN_entropy/bin/preprocessed_data/cora/edge_entropy.txt', "r").readlines()
    entropy_attentions_all=[]
    for line in edge_entropy_file:
        vector = [float(x) for x in line.strip('\n').strip(',').split(",")]
        entropy_attentions_all.append(vector)

    entropy_attentions_all=torch.from_numpy(np.array(entropy_attentions_all)).view(nodN*nodN,8).numpy()

    #entropy_attentions_all=torch.randn(nodN*nodN,8).numpy()

    #adj_citeseer=read_txt()
    adj_cora=read_csv()


    entropy_attentions_list=[]
    #entropy_attention_1=torch.zeros(nodN,nodN)
    for i in range(8):
        entropy_attention_i = torch.from_numpy(np.array(entropy_attentions_all[:, i])).float().view(nodN, nodN)
        #entropy_attention_1+=entropy_attention_i
        entropy_attention_i += adj_cora
        #no attention(gnn)
        #entropy_attention_i = torch.where(adj_cora > 0, entropy_attention_i, zero_vec)
        entropy_attentions_list.append(entropy_attention_i)

    #entropy_attention_1+=adj_cora
    #entropy_attentions_list.append(entropy_attention_1)

    return entropy_attentions_list

 #
    # entropy_attentions_list=[]
    # for i in range(8):
    #     attention=[]
    #     filename='../edGNN_entropy/bin/preprocessed_data/cora/attentions/attention_{}.txt'.format(i)
    #     attention_file=open(filename,"r").readlines()
    #     for line in attention_file:
    #         vector3 = [float(x) for x in line.strip('\n').strip(',').split(",")]
    #         attention.append(vector3)
    #     attention=torch.from_numpy(np.array(attention)).float().view(nodN,nodN).cuda()
    #     entropy_attentions_list.append(attention)

# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range

# for i in range(8):
#     entropy_attention_array=torch.from_numpy(entropy_attentions_all[:, i]).view(nodN,nodN).numpy()
#     entropy_attention=[]
#     for j in range(nodN):
#         entropy_attention.append(normalization(entropy_attention_array[j]))
#     entropy_attention=torch.from_numpy(np.array(entropy_attention)).float().cuda()
#     print('entropy_attention',entropy_attention.size())
#     entropy_attentions_list.append(entropy_attention)