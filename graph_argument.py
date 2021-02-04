import numpy as np
import torch
import random
import time
import copy
import torch.nn as nn

#Negative FeedBack Element-wise Disturb Node
def ngfb_disturb_node(opt,data,seed):
    nnode = data.num_nodes

    hop_np = torch.load(opt.hop_file).todense()
    
    hop_np[np.where(hop_np==0)] = np.max(hop_np)+10
    np.fill_diagonal(hop_np, 0)
    
    random.seed(seed)
    np.random.seed(seed) 

    adj = data.edge_index.numpy()
    new_adj = np.zeros((nnode,nnode),dtype='int8')
    new_adj[(adj[0],adj[1])]=1

    prob = data.contrastive_weight.numpy() 
    prob = pow(prob,opt.sharpen_time) 
    p_sum = np.sum(prob)
    prob = prob / p_sum

    prev_adj = copy.deepcopy(new_adj)
    norm_gap = 0
    
    while norm_gap<opt.fnorm_thre:

        center_node = np.random.choice(nnode, size=1, replace=False, p = prob)[0]
        temp_neb = new_adj[center_node]

        t_add = np.random.randint(opt.max_add_num) 
        t_rmv = np.random.randint(opt.max_rmv_num) 

        if t_add:
            where_none = np.where(temp_neb==0)[0]
            add_list = np.random.choice(where_none.size,size=t_add,replace=False)
            add_pos = where_none[add_list]
            for iter_x in add_pos:
            	new_adj[iter_x][center_node] = 1
            	new_adj[center_node][iter_x] = 1
            
        if t_rmv:
            where_edge = np.where(temp_neb==1)[0]
            if where_edge.size==0:continue
            rmv_list = np.random.choice(where_edge.size,size=t_rmv,replace=False)
            rmv_pos = where_edge[rmv_list]
            for iter_y in rmv_pos:
            	new_adj[iter_y][center_node] = 0
            	new_adj[center_node][iter_y] = 0

        data.embedding_mask[center_node]  = torch.ge(torch.rand(1,data.x.size(1)),opt.embedding_mask_rate).type(torch.BoolTensor).type(torch.FloatTensor)

        prob = modify_prob(prob,center_node,hop_np[center_node],(t_add + t_rmv)*1.0/(opt.max_add_num+opt.max_rmv_num))
        norm_gap =  np.linalg.norm((new_adj - prev_adj),ord='fro')


    where_new = np.where(new_adj>0)
    new_edge = [where_new[0],where_new[1]]
    new_edge_tensor = torch.from_numpy(np.array(new_edge))

    return new_edge_tensor,embedding_mask_tensor


def modify_prob(old_prob,center_node,node_neb, degree):
    hop_list = [1,2,3]
    decay_list = [0.1,0.3,0.5]
    decay_list = [x * (1 - degree) for x in decay_list]

    new_prob = copy.deepcopy(old_prob)
    new_prob[center_node] = 0
    for (iter_hop, iter_weight) in zip(hop_list,decay_list):
        where_node = np.where(node_neb==iter_hop)[0]
        if where_node.size==0: continue
        new_prob[where_node] *= iter_weight

    new_sum = np.sum(new_prob)
    new_prob = new_prob / new_sum
    return new_prob






