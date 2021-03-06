import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv



class StandGAT1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(StandGAT1, self).__init__()
        gat_para1 = 4
        self.conv1 = GATConv(nfeat, nclass,heads=gat_para1)
        self.dropout_p = dropout


    def forward(self, x, adj):
        edge_index = adj
        x = F.relu(self.conv1(x, edge_index))
        return x

  
class StandGAT2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(StandGAT2, self).__init__()
        gat_para1 = 4
        self.conv1 = GATConv(nfeat, nclass, heads=gat_para1)
        self.conv2 = GATConv(nhid,  nclass, heads=gat_para1)
       
        self.dropout_p = dropout

    def forward(self, x, adj):

        edge_index = adj

        x0 = F.relu(self.conv1(x, edge_index))

        x1 = F.dropout(x0, p= self.dropout_p, training=self.training)

        x2 = self.conv2(x1, edge_index)
        return x2

class StandGATX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(StandGATX, self).__init__()
        gat_para1 = 4
        self.conv1 = GATConv(nfeat, nclass, heads=gat_para1)
        self.conv2 = GATConv(nhid,  nclass, heads=gat_para1)
        self.convx = nn.ModuleList([GATConv(nhid, nhid, heads=gat_para1) for _ in range(nlayer-2)])
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

