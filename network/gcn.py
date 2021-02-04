import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class StandGCN1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(StandGCN1, self).__init__()
        self.conv1 = GCNConv(nfeat, nclass)
        self.dropout_p = dropout

    def forward(self, x, adj):
        edge_index = adj
        x = F.relu(self.conv1(x, edge_index))
    
        return x


class StandGCN2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(StandGCN2, self).__init__()
        
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.dropout_p = dropout

    def forward(self, x, adj):
        x_1 = self.conv1(x,adj)
        x1 = F.relu(x_1)
        x1 = F.dropout(x1, p= self.dropout_p, training=self.training)
        x_2 = self.conv2(x1, adj)

        return x_2


class StandGCNX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(StandGCNX, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nclass)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid) for _ in range(nlayer-2)])
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

