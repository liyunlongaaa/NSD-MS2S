# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def SoftCrossEntropy(inputs, target, reduction='mean'):
    '''
    inputs: Time * Num_class 
    target: Time * Num_class
    '''
    #print(inputs.shape)
    #print(target.shape)
    log_likelihood = -torch.nn.functional.log_softmax(inputs, dim=-1)
    batch = inputs.shape[0]
    if reduction == 'mean':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

def SoftCrossEntropy_4Targets(ypreds, label_data):
    loss = SoftCrossEntropy(ypreds[0], label_data[0]) + SoftCrossEntropy(ypreds[1], label_data[1]) + SoftCrossEntropy(ypreds[2], label_data[2]) + SoftCrossEntropy(ypreds[3], label_data[3])
    return loss

def CrossEntropy_SingleTargets(ypreds, label):
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0
    for i in range(len(ypreds)):
        loss += criterion(ypreds[i], label[i])
    return loss

def SoftCrossEntropy_SingleTargets(ypreds, label):
    loss = 0
    for i in range(len(ypreds)):
        loss += SoftCrossEntropy(ypreds[i], label[i])
    return loss

def Soft_BCE_SingleTargets(ypreds, label, reduction='mean', eps=1e-6):
    """
    ypreds: (B,N,T)
    label: (B,N,T)
    sigmoid or not
    """
    loss = 0
    probs = torch.sigmoid(ypreds)
    if reduction == 'mean':
        loss = -torch.mean(label * torch.log(probs + eps) + (1 - label) * torch.log(1 - probs + eps))
        #print(loss)
    else:
        loss = -(label * torch.log(y_preds + eps) + (1 - label) * torch.log(1 - y_preds + eps))
    
    return loss


class APLoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(APLoss, self).__init__()
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()
        self.count = 1
        print('Initialised AngleProto')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label   = torch.from_numpy(np.asarray(range(0,stepsize))).cuda()
        if self.count % 50 == 0 and torch.distributed.get_rank() == 0:
            print(cos_sim_matrix, self.w, self.b, self.w.grad, self.w.data)
        nloss   = self.criterion(cos_sim_matrix, label)
        self.count += 1
        return nloss


class PrototypicalLoss(nn.Module):

    def __init__(self, **kwargs):
        super(PrototypicalLoss, self).__init__()

        self.test_normalize = False

        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised Prototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2
        
        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        output  = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))**2)
        label   = torch.from_numpy(np.asarray(range(0,stepsize))).cuda()
        nloss   = self.criterion(output, label)

        return nloss


class GE2ELoss(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(GE2ELoss, self).__init__()

        self.test_normalize = True
        
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised GE2E')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        gsize = x.size()[1]
        centroids = torch.mean(x, 1)
        stepsize = x.size()[0]

        cos_sim_matrix = []

        for ii in range(0,gsize): 
            idx = [*range(0,gsize)]
            idx.remove(ii)
            exc_centroids = torch.mean(x[:,idx,:], 1)
            cos_sim_diag    = F.cosine_similarity(x[:,ii,:],exc_centroids)
            cos_sim         = F.cosine_similarity(x[:,ii,:].unsqueeze(-1),centroids.unsqueeze(-1).transpose(0,2))
            cos_sim[range(0,stepsize),range(0,stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim,1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix,dim=1)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        label = torch.from_numpy(np.asarray(range(0,stepsize))).cuda()
        nloss = self.criterion(cos_sim_matrix.view(-1,stepsize), torch.repeat_interleave(label,repeats=gsize,dim=0).cuda())
    
        return nloss