import numpy as np
import torch
import random
import time
import copy

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def index2dense(edge_index,nnode=2708):

    indx = edge_index.numpy()

    adj = np.zeros((nnode,nnode),dtype = 'int8')

    adj[(indx[0],indx[1])]=1

    new_adj = torch.from_numpy(adj).float()

    return new_adj

def index2adj(inf,nnode = 2708):

    indx = inf.numpy()

    print(nnode)

    adj = np.zeros((nnode,nnode),dtype = 'int8')

    adj[(indx[0],indx[1])]=1

    return adj
