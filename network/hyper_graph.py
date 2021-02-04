import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv


class HyperGraph1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(HyperGraph1, self).__init__()
        self.conv1 = HypergraphConv(nfeat, nclass)
        self.dropout_p = dropout


    def forward(self, x, adj):
        edge_index = adj
 
        x = F.relu(self.conv1(x, edge_index))
    
        return x

class HyperGraph2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(HyperGraph2, self).__init__()
        self.conv1 = HypergraphConv(nfeat, nhid)
        self.conv2 = HypergraphConv(nhid, nclass)

        self.dropout_p = dropout

      
    def forward(self, x, adj):
        edge_index = adj
        x0 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x0, p= self.dropout_p, training=self.training)
        x2 = self.conv2(x1, edge_index)
        return x2

class HyperGraphX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(HyperGraphX, self).__init__()
        self.conv1 = HypergraphConv(nfeat, nhid)
        self.conv2 = HypergraphConv(nhid, nclass)
        self.convx = nn.ModuleList([HypergraphConv(nhid, nhid) for _ in range(nlayer-2)])
        self.dropout_p = dropout

  
    def forward(self, x, adj):
        edge_index = adj

        x = F.relu(self.conv1(x, edge_index))

        for iter_layer in self.convx:
            x = F.dropout(x, p= self.dropout_p, training=self.training)
            x = F.relu(iter_layer(x, edge_index))

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)

        return x