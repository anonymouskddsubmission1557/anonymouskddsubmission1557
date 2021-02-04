import torch
import torch.nn as nn
import torch.nn.functional as F

from network.gcn import StandGCN1,StandGCN2,StandGCNX
from network.gat import StandGAT1,StandGAT2,StandGATX
from network.cheb_gcn import ChebGCN1,ChebGCN2,ChebGCNX
from network.arma import ARMA1, ARMA2, ARMAX
from network.hyper_graph import HyperGraph1,HyperGraph2,HyperGraphX
from network.stand_graph import StandGraph1,StandGraph2,StandGraphX
from network.feast import FeaST1,FeaST2,FeaSTX
from network.sage import GraphSAGE1,GraphSAGE2,GraphSAGEX
from network.mlp import MLP1,MLP2,MLPX


def get_model(opt):

    nfeat = opt.num_feature
    nclass = opt.num_class
    nhid = opt.num_hidden
    nlayer = opt.num_layer
    cuda = opt.gpu>-1
    dropout = opt.dropout
    
    model_opt = opt.model

    model_dict = {
        'gcn' : [StandGCN1,StandGCN2,StandGCNX],
        'gat' : [StandGAT1,StandGAT2,StandGATX],
        'cheb' : [ChebGCN1,ChebGCN2,ChebGCNX],
        'arma' : [ARMA1, ARMA2, ARMAX],
        'hyper' : [HyperGraph1,HyperGraph2,HyperGraphX],
        'stand' : [StandGraph1,StandGraph2,StandGraphX],
        'feast' : [FeaST1,FeaST2,FeaSTX],
        'sage' : [GraphSAGE1,GraphSAGE2,GraphSAGEX],
        'mlp' : [MLP1,MLP2,MLPX]
    }
    model_list = model_dict[model_opt]
    
    if nlayer==1:
        model = model_list[0](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)

    elif nlayer ==2:
        model = model_list[1](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)

    else:
        model = model_list[2](nfeat=nfeat,
                nhid=nhid,
                nclass=nclass,
                dropout=dropout,
                nlayer = nlayer)  

    if cuda: model.cuda()
    return model
