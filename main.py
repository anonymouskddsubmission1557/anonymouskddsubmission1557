import copy
import argparse
import os
import random
import sys,time
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score,accuracy_score

from models import get_model
from utils import set_seed,index2dense
from load_data import load_processed_data
from graph_argument import ngfb_disturb_node


def get_opt():
    parser = argparse.ArgumentParser()

    # GNN
    parser.add_argument('--model', default='gcn', type=str)
    parser.add_argument('--num-hidden', default=128, type=int)
    parser.add_argument('--num-feature', default=745, type=int)
    parser.add_argument('--num-class', default=7, type=int)
    parser.add_argument('--num-layer', default=2, type=int)


    # Training
    parser.add_argument('--lr', default=0.0075, type=float)
    parser.add_argument('--lr-decay-epoch', default=30, type=int)
    parser.add_argument('--lr-decay-rate', default=0.95, type=float)
    parser.add_argument('--weight-decay', default=0.0005, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--data-path',  default='', type=str)
    parser.add_argument('--hop-file',   default='', type=str)
    parser.add_argument('--data-name',  default='', type=str)

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--least-epoch', default=30, type=int)
    parser.add_argument('--early-stop', default=20, type=int)

    # Running
    parser.add_argument('--run-split-num', type=int, default=20,help='run N different split times')
    parser.add_argument('--run-init-num',  type=int, default=5, help='run N different init seeds')
    parser.add_argument('--shuffle-seed-list', default=[i for i in range(20)], type=list)
    parser.add_argument('--initial-seed-list', default=[i for i in range(5)],  type=list)

    #topology pertubation 
    parser.add_argument('--sharpen-time', default=0.25,  type=int)
    parser.add_argument('--fnorm-thre',   default=15, type=float, help='the Frobenius norm threshold. This is used to control the degree of the graph argumentation')
    parser.add_argument('--max-add-num',  default=10,  type=int)
    parser.add_argument('--max-rmv-num',  default=3,  type=int)
    parser.add_argument('--embedding-mask-rate', default=0.15, type=float)
    
    #cl
    parser.add_argument('--cl-loss-weight', default=1, type=float)
    parser.add_argument('--cl-confidence-thresh', default=0.55, type=float)
    parser.add_argument('--cl-softmax-temp', default=1, type=float)

    #group pagerank 
    parser.add_argument('--pagerank-prob', default=0.85, type=float,help="probility of going down instead of going back to the labeled node in the random walk")
    parser.add_argument('--reweight-node','-rw', action='store_true')
    parser.add_argument('--base-weight', default=1, type=float,help="the base weight of gcl schedule")
    parser.add_argument('--scale-weight', default=2, type=float,help="the scale weight of gcl schdule")
    parser.add_argument('--punish-weight', default=0.1, type=float, help="weight of other gpr value in correction max gpr value")

    #hard_negative 
    parser.add_argument('--dist-metric', default="cosine", type=str)
    parser.add_argument('--hard-weight', default=[0.75,1,0.5], type=float,nargs='+')
    parser.add_argument('--post-weight', default=0.5, type=float,help="weight of postive pairs in KL Loss")
    parser.add_argument('--negt-weight', default=0.25, type=float,help="weight of negtive pairs in KL Loss")
    parser.add_argument('--extra-dict', default = [0,50,150,500],type=int,nargs='+')

    opt = parser.parse_args()
    opt.data_path = opt.data_path + opt.data_name

    return opt

def train(model,opt,data,adj):

    cls_loss_func = torch.nn.CrossEntropyLoss()

    unsup_criterion = torch.nn.KLDivLoss(reduction='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)

    best_acc = 0
    best_epoch = 0

    for epoch in range(1, opt.epoch+1):
        
        if epoch > opt.lr_decay_epoch:
            new_lr = opt.lr * pow(opt.lr_decay_rate,(epoch-opt.lr_decay_epoch))
            new_lr = max(new_lr,1e-4)
            for param_group in optimizer.param_groups: param_group['lr'] = new_lr

        model.train()
        optimizer.zero_grad()

        sup_logits = model(data.x.cuda(), data.norm_adj.cuda())
        cls_loss = cls_loss_func(sup_logits[data.train_mask], data.y[data.train_mask].cuda())
        loss = cls_loss 

        if opt.cl_loss_weight!=0:
            with torch.no_grad():
                ori_logits_all = model(data.x.cuda(),adj.cuda())
                ori_prob_all = F.softmax(ori_logits_all,dim=-1) 
                ori_logits = ori_logits_all[data.unsuper_mask]


                if opt.cl_confidence_thresh != -1:
                    unsup_loss_mask = torch.max(ori_prob,dim=-1)[0]>cl_confidence_thresh
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len(ori_logits),dtype=torch.float32)

                unsup_loss_mask = unsup_loss_mask.cuda()

            cl_softmax_temp = opt.cl_softmax_temp if opt.cl_softmax_temp>0 else 1
            data.aug_adj = ngfb_disturb_node(opt,data,epoch)
            aug_logits = model(data.x.cuda() * data.embedding_mask.cuda(), data.aug_adj.cuda())
            aug_log_prob = F.log_softmax(aug_logits/cl_softmax_temp,dim=-1)

            self_loss = unsup_criterion(aug_log_prob[data.unsuper_mask],ori_prob)
            self_loss = torch.sum(self_loss,dim=-1)

            _, post_size, _, negt_size = opt.extra_dict
            node_size,dim_size = aug_logits.size()
            unlab_size, _ = ori_logits.size()
            
            unsup_loss = self_loss 
            
            if opt.post_weight !=0:
                post_ori_prob = ori_prob_all[data.post_cand].view(node_size,post_size,-1)
                post_ori_prob = post_ori_prob[data.unsuper_mask].contiguous()

                post_aug_prob = aug_log_prob[data.unsuper_mask]
                post_aug_prob = post_aug_prob.unsqueeze(1).expand(unlab_size,post_size,dim_size)

                post_loss = unsup_criterion(post_aug_prob,post_ori_prob)
                post_loss = torch.sum(post_loss,dim=-1)
                post_mean_loss = post_loss.mean(dim=-1) 

                unsup_loss +=  opt.post_weight * post_mean_loss

            if opt.negt_weight !=0:
                negt_ori_prob = ori_prob_all[data.negt_cand].view(node_size,negt_size,-1)
                negt_ori_prob = negt_ori_prob[data.unsuper_mask].contiguous()

                negt_aug_prob = aug_log_prob[data.unsuper_mask]
                negt_aug_prob = negt_aug_prob.unsqueeze(1).expand(unlab_size,negt_size,dim_size)

                negt_loss = unsup_criterion(negt_aug_prob,negt_ori_prob)
                negt_loss = torch.sum(negt_loss,dim=-1)
                negt_mean_loss = negt_loss.mean(dim=-1) 

                unsup_loss +=  opt.negt_weight * negt_mean_loss
          
            if opt.reweight_node:
                contrastive_weight = data.contrastive_weight[data.unsuper_mask].cuda()
                unsup_loss = torch.sum(unsup_loss * unsup_loss_mask * contrastive_weight,dim=-1) / torch.max(torch.sum(unsup_loss_mask,dim=-1),torch.tensor(1.).cuda())
            else:
                unsup_loss = torch.sum(unsup_loss * unsup_loss_mask,dim=-1) / torch.max(torch.sum(unsup_loss_mask,dim=-1),torch.tensor(1.).cuda())

            loss += unsup_loss * opt.cl_loss_weight
        
        loss.backward()
        
        optimizer.step()
        
        train_loss = loss / data.train_mask.size(0)
        print(train_loss)

        val_acc = test(opt,model,data,adj,data.valid_mask)

        if val_acc>best_acc:
            best_model = copy.deepcopy(model)
            best_acc = val_acc
            best_epoch = epoch 

        if opt.early_stop>0 and epoch - best_epoch > opt.early_stop: break

    print('best_epoch,best_val_acc:%d, %.4f'%(best_epoch,best_acc))

    return best_model


def test(opt,model,data,adj,target_mask):
    model.eval()

    target=data.y[target_mask].numpy()

    with torch.no_grad():
        out = model(data.x.cuda(), data.norm_adj.cuda())
    
    pred=out[target_mask].cpu().max(1)[1].numpy()

    acc = accuracy_score(target, pred) if pred.sum() > 0 else 0
    
    return acc


def main():

    opt = get_opt()
    if opt.gpu > -1: torch.cuda.set_device(opt.gpu)

    run_time_result = [[] for _ in range(opt.run_split_num)]

    all_list = []

    for iter_split_seed in range(opt.run_split_num):
 
        target_data = load_processed_data(opt,opt.data_path,opt.data_name,shuffle_seed = opt.shuffle_seed_list[iter_split_seed])
        setattr(opt, 'num_feature', target_data.num_features)
        setattr(opt, 'num_class', target_data.num_classes)

        adj = target_data.edge_index
            
        for iter_init_seed in range(opt.run_init_num):

            set_seed(seed_list[iter_init_seed], opt.gpu>-1)
            model = get_model(opt)

            best_model = train(model,opt,target_data,adj)

            test_acc = test(opt,best_model, target_data,adj,target_data.test_mask,'test')

            test_acc = round(test_acc,4)

            print(test_acc)


if __name__ == '__main__':
    main()