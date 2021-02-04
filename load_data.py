import torch
import numpy as np
import random
import codecs
import copy
import os,sys

from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets.planetoid import Planetoid

from group_pr import get_gpr,get_npr,get_tig,cl_weight_schedule
from hard_neg import sample_hard_negative



def get_split(all_idx,all_label,train_each=20,valid_each=30,nclass = 7):

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx = []
    
    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < train_each:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==train_each*nclass:break

    assert sum(train_list)==train_each*nclass
    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==valid_each*nclass:break

    assert sum(valid_list)==valid_each*nclass
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node



def load_processed_data(opt,data_path,data_name,shuffle_seed = -1, train_each = 20, valid_each = 30, label_save_path = ''):
    data_dict = {'cora':'planetoid','citeseer':'planetoid','pubmed':'planetoid',
                'cs':'coauthor','phy':'coauthor'}
    
    target_type = data_dict[data_name]
    if target_type == 'amazon':
        target_dataset = Amazon(data_path, name=data_name)
    elif target_type == 'planetoid':
        target_dataset = Planetoid(data_path, name=data_name)

    target_data=target_dataset[0]
    target_data.num_classes = np.max(target_data.y.numpy())+1

    target_data.train_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.valid_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.test_mask  = torch.zeros(target_data.num_nodes, dtype=torch.bool)
    target_data.unsuper_mask = torch.zeros(target_data.num_nodes, dtype=torch.bool)

    mask_list = [i for i in range(target_data.num_nodes)]
    
    random.seed(shuffle_seed)
    random.shuffle(mask_list)
    train_mask_list,valid_mask_list,test_mask_list,train_cls_node = get_split(mask_list,target_data.y.numpy(),train_each,valid_each, nclass=target_data.num_classes)

    
    for iter0 in range(target_data.num_nodes):
        target_data.unsuper_mask[iter0] = 1
    for iter1 in train_mask_list:
        target_data.train_mask[iter1]=1
        target_data.unsuper_mask[iter1] = 0
    for iter2 in valid_mask_list:
        target_data.valid_mask[iter2]=1
    for iter3 in test_mask_list:
        target_data.test_mask[iter3]=1

    target_data.gpr = get_gpr(target_data.edge_index,np.array(train_cls_node),target_data.num_nodes,opt.pagerank_prob)
    target_data.npr = get_npr(target_data.x,train_cls_node)

    target_data.gpr = target_data.gpr * target_data.npr
    target_data.tig = get_tig(target_data.gpr , opt.punish_weight)

    target_data.contrastive_weight = cl_weight_schedule(target_data.num_nodes,opt.base_weight,opt.scale_weight,target_data.tig)

    sample_hard_negative(opt,target_data)

    target_data.embedding_mask = torch.ones(target_data.x.size()).type(torch.FloatTensor)
    
    return target_data


















