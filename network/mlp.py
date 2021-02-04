import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=1):
        super(MLP1, self).__init__()
        self.conv1 = nn.Linear(nfeat, nclass)
        self.dropout_p = dropout

    def forward(self, x, adj):

        edge_index = adj
        x = F.relu(self.conv1(x))
    
        return x

class MLP2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=2):
        super(MLP2, self).__init__()
        
        self.conv1 = nn.Linear(nfeat, nhid)
        self.conv2 = nn.Linear(nhid, nclass)
        self.conv3 = nn.Linear(nhid, nhid)

        self.dropout_p = dropout

    def forward(self, x, adj):

        edge_index = adj

        x0 = F.relu(self.conv1(x))

        x1 = F.dropout(x0, p= self.dropout_p, training=self.training)

        x2 = self.conv2(x1)
        
        return x2



class MLPX(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,nlayer=3):
        super(MLPX, self).__init__()
        self.conv1 = nn.Linear(nfeat, nhid)
        self.conv2 = nn.Linear(nhid, nclass)
        self.convx = nn.ModuleList([nn.Linear(nhid, nhid) for _ in range(nlayer-2)])
        self.dropout_p = dropout


    def forward(self, x, adj):
        edge_index = adj

        x = F.relu(self.conv1(x))

        for iter_layer in self.convx:
            x = F.dropout(x, p= self.dropout_p, training=self.training)
            x = F.relu(iter_layer(x))

        x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x)

        return x

