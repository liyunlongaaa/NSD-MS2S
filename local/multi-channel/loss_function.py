# -*- coding: utf-8 -*-

import torch


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