from __future__ import division

import copy
import math

import scipy as sp
import scipy.sparse as sprs
import scipy.spatial
import scipy.sparse.linalg
import numpy as np

import torch
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

from utils import index2adj


#Calculates PageRank given a csr graph by solving the linera algebra
def group_pagerank_alg(A, personalize, p=0.85):
    cls_size = personalize.shape[1]

    n, _ = A.shape
    r = sp.asarray(A.sum(axis=1)).reshape(-1) 

    k = r.nonzero()[0]

    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    s = (personalize / personalize.sum(axis=0))*n 

    I = sprs.eye(n)
    x = sprs.linalg.spsolve((I - p * A.T @ D_1), s)

    x = x / x.max(axis=0) 
    return x

#Calculates PageRank given a csr graph using the iterative convergence
def group_pagerank_power(A, personalize, p=0.85, max_iter=100,tol=1e-06):
    
    cls_size = personalize.shape[1]

    n, _ = A.shape
    r = sp.asarray(A.sum(axis=1)).reshape(-1)

    k = r.nonzero()[0]

    D_1 = sprs.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    s = (personalize / personalize.sum(axis=0))*n

    z_T = (((1 - p) * (r != 0) + (r == 0)) / n)
    z_T = np.tile(z_T,(cls_size,1))

    W = p * A.T @ D_1

    x = copy.deepcopy(s)

    oldx = np.ones(x.shape)

    iteration = 0

    gap = tol + 10;

    while  gap > tol:

        print("[%d]: %.4f"%(iteration,gap))

        oldx = copy.deepcopy(x)
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
        gap = sp.linalg.norm(x - oldx)


    x = x / x.max(axis=0)

    return x

def get_gpr(edge_index, labeled_node, nnode, p=0.85):

    cls_size = labeled_node.shape[0]

    dense_adj = index2adj(edge_index,nnode)

    sparse_adj = sprs.csr_matrix(dense_adj)

    group_init = np.zeros((nnode, cls_size))

    for i in range(cls_size):
        for j in labeled_node[i]:
            group_init[j][i] = 1.0

    return group_pagerank_alg(sparse_adj, group_init, p)

#compute the similarity with prototype of labeled node
def get_npr(x_tensor,labeled_node):
    nnode,nfeat = x_tensor.size()

    node_id_np = np.array(labeled_node)
    nclass,neach = node_id_np.shape

    node_id_all = node_id_np.reshape(-1)
    node_id_all = torch.from_numpy(node_id_all).type(torch.LongTensor)

    node_fe_all = x_tensor[node_id_all].contiguous().view(nclass,neach,-1)
    node_prot = torch.mean(node_fe_all,dim=1).squeeze()

    node_prot = node_prot.numpy()
    node_all  = x_tensor.numpy()

    prot_simi = cosine_similarity(node_all,node_prot)
    prot_simi = (prot_simi + 1) / 2 
    return prot_simi


#compute the tig values
def get_tig(pr_matrix,punish_weight):

    pr_max = np.max(pr_matrix,axis=1)
    pr_sum = np.sum(pr_matrix,axis=1)

    nnode = pr_matrix.shape[0]
    cls_size = pr_matrix.shape[1]

    pr_other = pr_sum - pr_max
    pr_other = pr_other / (cls_size - 1)

    tig = pr_max - punish_weight * pr_other

    return tig

def cl_weight_schedule(nnode,base_w,scale_w,tig_array):

    tig_list = tig_array.tolist()

    id2tig = {i:tig_list[i] for i in range(len(tig_list))}

    sorted_tig = sorted(id2tig.items(),key=lambda x:x[1])

    id2rank = {sorted_tig[i][0]:i for i in range(nnode)}

    tig_rank = [id2rank[i] for i in range(nnode)]

    cl_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*math.pi/nnode))) for x in tig_rank]

    cl_weight_tensor = torch.from_numpy(np.array(cl_weight)).type(torch.FloatTensor)

    return cl_weight_tensor



