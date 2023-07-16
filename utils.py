# coding:utf-8
from collections import defaultdict
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph
from scipy import sparse as sp
from sklearn.decomposition import PCA
import numpy as np
import networkx as nx



def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10

def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

def create_pairs_dict(pairs):
    pairs_dict = {}
    for x,y in pairs:
        if x not in pairs_dict.keys():
            pairs_dict[x] = [y]
        else:
            pairs_dict[x].append(y)
    return pairs_dict

def construct_graph(adatas,match,n=5):
    cell_index = 0
    id_grp = []
    for i in range(len(adatas)):
        adj, _ = get_adj(adatas[i].X, k=n)
        adj = coo_matrix(adj)
        fake = np.array([-1] * cell_index)
        find = np.concatenate((fake, np.array(range(len(adatas[i].X))))).flatten()
        id_grp1 = np.array([
            np.concatenate((np.where(find == adj.row[i])[0],
                            np.where(find == adj.col[i])[0]))
            for i in range(adj.nnz)
        ])
        id_grp2 = np.array([
            np.concatenate((np.where(find == adj.col[i])[0],
                            np.where(find == adj.row[i])[0]))
            for i in range(adj.nnz)
        ])
        id_grp.append(id_grp1)
        id_grp.append(id_grp2)
        cell_index += len(adatas[i].X)

    matrix = np.identity(cell_index)
    for i in range(len(id_grp)):
        matrix[tuple(id_grp[i].T)] = 1

    matrix[tuple(match.T)] = 1
    matrix[tuple(match[:, [1, 0]].T)] = 1
    A = graph(matrix)
    A = nx.adjacency_matrix(nx.from_dict_of_lists(A))
    adj = A.toarray()
    adj_n = norm_adj(adj)

    return adj,adj_n


def graph(matrix):
    adj = defaultdict(list)
    for i, row in enumerate(matrix):
        for j, adjacent in enumerate(row):
            if adjacent:
                adj[i].append(j)
        if adj[i].__len__ == 0:
            adj[i] = []
    return adj

def get_adj(count, k=10, pca=30, mode="connectivity"):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    A = kneighbors_graph(countp, k, mode=mode, metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n





