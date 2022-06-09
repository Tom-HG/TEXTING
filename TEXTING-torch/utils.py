import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import re
from tqdm import tqdm
# import sparse

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'allx_adj', 'allx_embed', 'ally']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, tx_adj, tx_embed, ty, allx_adj, allx_embed, ally = tuple(objects)
    train_adj = []
    train_embed = []
    val_adj = []
    val_embed = []
    test_adj = []
    test_embed = []

    for i in range(len(y)):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        train_adj.append(adj)
        train_embed.append(embed)

    for i in range(len(y), len(ally)):  # train_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        val_adj.append(adj)
        val_embed.append(embed)

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        test_adj.append(adj)
        test_embed.append(embed)

    train_adj = np.array(train_adj, dtype=object)
    val_adj = np.array(val_adj, dtype=object)
    test_adj = np.array(test_adj, dtype=object)
    train_embed = np.array(train_embed, dtype=object)
    val_embed = np.array(val_embed, dtype=object)
    test_embed = np.array(test_embed, dtype=object)
    train_y = np.array(y)
    val_y = np.array(ally[len(y):len(ally)]) # train_size])
    test_y = np.array(ty)

    return train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y


def load_data_bert(dataset_str):
    names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'val_adj', 'val_embed', 'valy']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, tx_adj, tx_embed, ty, val_adj, val_embed, valy = tuple(objects)
    train_adj = []
    # train_embed = []
    dev_adj = []
    # val_embed = []
    test_adj = []
    # test_embed = []

    for i in range(len(y)):
        adj = x_adj[i].toarray()
        # embed = np.array(x_embed[i])
        train_adj.append(adj)
        # train_embed.append(embed)
    
    train_embed = tokenizer(x_embed, padding=True, truncation=True, return_tensors="pt")
    # print(np.shape(x_adj))
    # print('train_embed')
    # print(train_embed['input_ids'].size())
    # print()
    # print(np.shape(train_adj))


    # print(len(valy))
    # print(len(val_adj))
    for i in range(len(valy)):  # train_size):
        adj = val_adj[i].toarray()
        # embed = np.array(allx_embed[i])
        dev_adj.append(adj)
        # val_embed.append(embed)
    dev_embed = tokenizer(val_embed, padding=True, truncation=True, return_tensors="pt")

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        # embed = np.array(tx_embed[i])
        test_adj.append(adj)
        # test_embed.append(embed)
    # print(tx_embed)
    test_embed = tokenizer(tx_embed, padding=True, truncation=True, return_tensors="pt")

    train_adj = np.array(train_adj, dtype=object)
    dev_adj = np.array(dev_adj, dtype=object)
    test_adj = np.array(test_adj, dtype=object)
    # train_embed = np.array(train_embed, dtype=object)
    # val_embed = np.array(val_embed, dtype=object)
    # test_embed = np.array(test_embed, dtype=object)
    train_y = np.array(y)
    dev_y = np.array(valy) # train_size])
    test_y = np.array(ty)

    return train_adj, train_embed, train_y, dev_adj, dev_embed, dev_y, test_adj, test_embed, test_y


def load_data_v2(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'allx_adj', 'allx_embed', 'ally']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x_adj, x_embed, y, tx_adj, tx_embed, ty, allx_adj, allx_embed, ally = tuple(objects)
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    # print(np.shape(allx_adj))
    # print(np.shape(allx_embed))
    # print(np.shape(ally))

    train_adj = [[],[],[]]
    train_embed = []
    val_adj = [[],[],[]]
    val_embed = []
    test_adj = [[],[],[]]
    test_embed = []

    for i in range(len(y)):
        embed = np.array(x_embed[i])
        train_embed.append(embed)

    for j in range(3):
        for i in range(len(y)):
            adj = x_adj[j][i].toarray()
            train_adj[j].append(adj)


    for i in range(len(y), len(ally)): #train_size):
        embed = np.array(allx_embed[i])
        val_embed.append(embed)

    for j in range(3):
        for i in range(len(y), len(ally)): #train_size):
            adj = allx_adj[j][i].toarray()
            val_adj[j].append(adj)

    for i in range(len(ty)):
        embed = np.array(tx_embed[i])
        test_embed.append(embed)

    for j in range(3):
        for i in range(len(ty)):
            adj = tx_adj[j][i].toarray()
            test_adj[j].append(adj)

    train_adj = np.array(train_adj, dtype=object)
    val_adj = np.array(val_adj, dtype=object)
    test_adj = np.array(test_adj, dtype=object)
    train_embed = np.array(train_embed, dtype=object)
    val_embed = np.array(val_embed, dtype=object)
    test_embed = np.array(test_embed, dtype=object)
    train_y = np.array(y)
    val_y = np.array(ally[len(y):len(ally)]) # train_size])
    test_y = np.array(ty)

    return train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def coo_to_tuple(sparse_coo):
    return (sparse_coo.coords.T, sparse_coo.data, sparse_coo.shape)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    max_length = max([len(f) for f in features])
    
    for i in tqdm(range(features.shape[0])):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0] # padding for each epoch
        feature = np.pad(feature, ((0,pad),(0,0)), mode='constant')
        features[i] = feature
    
    return np.array(list(features))


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    mask = np.zeros((adj.shape[0], max_length, 1)) # mask for padding

    for i in tqdm(range(adj.shape[0])):
        adj_normalized = normalize_adj(adj[i]) # no self-loop (bert移除)
        # adj_normalized = adj[i]
        pad = max_length - adj_normalized.shape[0] # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
        mask[i,:adj[i].shape[0],:] = 1.
        adj[i] = adj_normalized

    return np.array(list(adj)), mask # coo_to_tuple(sparse.COO(np.array(list(adj)))), mask


def preprocess_adj_bert(adj, length):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    # max_length = max([a.shape[0] for a in adj])
    max_length = length
    mask = np.zeros((adj.shape[0], max_length, 1)) # mask for padding

    adj_bert = list()
    for i in tqdm(range(adj.shape[0])):
        # adj_normalized = normalize_adj(adj[i]) # no self-loop (bert移除)
        adj_normalized = adj[i]
        pad = max_length - adj_normalized.shape[0] # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
        mask[i,:adj[i].shape[0],:] = 1.
        adj_bert.append(adj_normalized)

    # return np.array(list(adj)), mask # coo_to_tuple(sparse.COO(np.array(list(adj)))), mask
    return np.array(adj_bert), mask

def construct_feed_dict(features, support, mask, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['mask']: mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = clean_str(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"，。！？·,", " ", string)
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


import datetime


def print_log(msg='', end='\n'):
    now = datetime.datetime.now()
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)

    if isinstance(msg, str):
        lines = msg.split('\n')
    else:
        lines = [msg]

    for line in lines:
        if line == lines[-1]:
            print('[' + t + '] ' + str(line), end=end)
        else:
            print('[' + t + '] ' + str(line))