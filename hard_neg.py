import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix
import random
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F



#opt: all config info
#data: all data info
#return: [N * K], n is the node size and K is the negative size;
#this func will first selecet the most hardest nodes set first and then conduct the mixing operation
def sample_hard_negative(opt,data):
	nnode = data.num_nodes
	nfeat = data.num_features
	nclass = data.num_classes

	epsilon = 1e-8

	#selecet the most hardest nodes set around the node based on both gpr KL div, node feature simi and least jump hop
	
	##1/3 feature simi matrix
	feat_dist = pairwise_distances(data.x, data.x, metric=opt.dist_metric) 
	feat_matrix = torch.from_numpy(feat_dist).type(torch.FloatTensor)

	##2/3 gpr kl matrix
	gprt = data.gpr + epsilon
	gprt = torch.from_numpy(gprt).type(torch.FloatTensor)
	gprt = F.softmax(gprt,dim=1)

	gpr1 = gprt.expand(nnode,nnode,nclass)
	gpr2 = gpr1.transpose(0,1)

	gpr1 = gpr1.contiguous().view(-1,nclass)
	gpr2 = gpr2.contiguous().view(-1,nclass)

	gpr_kl = torch.sum(gpr1 * torch.log2(gpr1/gpr2),dim=-1)
	gpr_matrix = gpr_kl.view(nnode,nnode) 

	##3/3 hop matridx,rely on the adj hop matrix (how many hops needs at least to jump from a to b)
	hop_np = torch.load(opt.hop_file).todense()
	hop_np[np.where(hop_np==0)] = np.max(hop_np)+10
	np.fill_diagonal(hop_np, 0)
	hop_matrix = torch.from_numpy(hop_np).type(torch.FloatTensor)

	#consider all the 3 matrix to filter the hard candidate
	k1,k2,k3 = opt.hard_weight
	all_matrix = k1 * (feat_matrix / feat_matrix.max()) + k2 * (gpr_matrix / gpr_matrix.max()) + k3 * (hop_matrix / hop_matrix.max()) + 100 *torch.eye(nnode)#mask the node itself

	#cand_idx: [node_size * cand * size] 
	cand_val,cand_idx = torch.topk(all_matrix,nnode,dim=-1,largest=False,sorted=True)

	post_beg, post_size, negt_beg, negt_size = opt.extra_dict
	data.post_cand = cand_idx[:,post_beg : post_beg + post_size].contiguous().view(-1)
	data.negt_cand = cand_idx[:,negt_beg : negt_beg + negt_size].contiguous().view(-1)


