import copy
import os.path as osp
from typing import Optional

import torch
from tqdm import tqdm
from torch_sparse import SparseTensor
import torch.nn.functional as F
import numpy as np
import sys


class GraphSAINTSampler(torch.utils.data.DataLoader):
 
    def __init__(self, data, batch_size: int, num_steps: int = 1,
                 sample_coverage: int = 0, save_dir: Optional[str] = None,
                 log: bool = True, **kwargs):

        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data

        self.num_steps = num_steps
        self.__batch_size__ = batch_size
        self.sample_coverage = sample_coverage
        self.log = log

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(N, N))

        self.data = copy.copy(data)
        self.data.edge_index = None

        super(GraphSAINTSampler,
              self).__init__(self, batch_size=1, collate_fn=self.__collate__,
                             **kwargs)

        if self.sample_coverage > 0:
            path = osp.join(save_dir or '', self.__filename__)
            if save_dir is not None and osp.exists(path):  # pragma: no cover
                self.node_norm, self.edge_norm = torch.load(path)
            else:
                self.node_norm, self.edge_norm = self.__compute_norm__()
                if save_dir is not None:  # pragma: no cover
                    torch.save((self.node_norm, self.edge_norm), path)

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self.sample_coverage}.pt'

    def __len__(self):
        return self.num_steps

    def __sample_nodes__(self, batch_size):
        raise NotImplementedError

    def __getitem__(self, idx):
        node_idx = self.__sample_nodes__(self.__batch_size__).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)
        return node_idx, adj

    def __collate__(self, data_list):
        assert len(data_list) == 1
        node_idx, adj = data_list[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if isinstance(item, torch.Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        if self.sample_coverage > 0:
            data.node_norm = self.node_norm[node_idx]
            data.edge_norm = self.edge_norm[edge_idx]

        return data

    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        loader = torch.utils.data.DataLoader(self, batch_size=200,
                                             collate_fn=lambda x: x,
                                             num_workers=self.num_workers)

        if self.log:  
            pbar = tqdm(total=self.N * self.sample_coverage)
            pbar.set_description('Compute GraphSAINT normalization')

        num_samples = total_sampled_nodes = 0
        while total_sampled_nodes < self.N * self.sample_coverage:
            for data in loader:
                for node_idx, adj in data:
                    edge_idx = adj.storage.value()
                    node_count[node_idx] += 1
                    edge_count[edge_idx] += 1
                    total_sampled_nodes += node_idx.size(0)

                    if self.log:  
                        pbar.update(node_idx.size(0))
            num_samples += 200

        if self.log:  
            pbar.close()

        row, _, edge_idx = self.adj.coo()
        t = torch.empty_like(edge_count).scatter_(0, edge_idx, node_count[row])
        edge_norm = (t / edge_count).clamp_(0, 1e4)
        edge_norm[torch.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N

        return node_norm, edge_norm



class SAINTTIFASampler(GraphSAINTSampler):

    def __init__(self, data, link_dict, sharpen_time: int,batch_size: int, walk_length: int,
                 num_steps: int = 1, sample_coverage: int = 0, save_dir: Optional[str] = None, log: bool = True, **kwargs):
        
        self.walk_length = walk_length
        super(SAINTTIFASampler,self).__init__(data, batch_size, num_steps, sample_coverage, save_dir, log, **kwargs)

        self.link_dict = link_dict

        self.p = []
        
        self.negb_link = link_dict['negb_link'] # adjacency list with padding
        prob_np = link_dict['prob_link'] # kl matrix for adjacency list
        
        where_pad = np.where(prob_np==0)
        
        prob_np = 1 / (prob_np + 1e-12) 
        prob_np = pow(prob_np,sharpen_time) # sharpen the prob distribution

        prob_np[where_pad] = float('-inf') 
        prob = torch.from_numpy(prob_np).type(torch.FloatTensor)

        prob = F.softmax(prob,dim=-1)
        self.prob_link = prob.numpy()

    @property
    def __filename__(self):
        return (f'{self.__class__.__name__.lower()}_{self.walk_length}_'
                f'{self.sample_coverage}.pt')

    def __sample_nodes__(self, batch_size):
        start = torch.randint(0, self.N, (batch_size, ), dtype=torch.long)

        select_node = []

        for x in start.tolist():
            x_list = [x]

            for _ in range(self.walk_length):
                x_nebg = self.negb_link[x]
                x_prob = self.prob_link[x]

                x = np.random.choice(x_nebg,1,p=x_prob)
                x_list.append(x)

            select_node.append(x_list)

        node_idx = torch.from_numpy(np.array(select_node))
        return node_idx.view(-1)