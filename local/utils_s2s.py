# -*- coding: utf-8 -*-

import torch
import logging
import sys
import pdb
from typing import Dict, List, Tuple
from torch.nn import Module, ModuleList
import copy, os
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def save_checkpoint(model, optimizer, filename):
    try:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'configs': model.configs}, filename)
    except:
        torch.save({'model': model.state_dict(), \
            'optimizer_tsvad': optimizer['tsvad'].state_dict(), \
            'optimizer_resnet': optimizer['resnet'].state_dict(), 'configs': model.configs},  filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        
def load_checkpoint_join_training(model, optimizer, filename):
    checkpoint = torch.load(filename)
    # pdb.set_trace()
    if model is not None:
        model_dict = model.state_dict()
        # pdb.set_trace()
        state_dict_2 = {k:v for k,v in checkpoint['model'].items()}
        # pdb.set_trace()
        model_dict.update(state_dict_2)
        model.load_state_dict(model_dict)
        # model_dict['FC.2.weight'] - checkpoint['model']['FC.2.weight']
        # pdb.set_trace()
        # model.load_state_dict(checkpoint['model'])
    # pdb.set_trace()
    if optimizer is not None and 'join_train' in filename:
        print('load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])

def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s', datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename)) 
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger

def average_states(
    states_list: List[Dict[str, torch.Tensor]]
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key]

    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state

def parse_epochs(string: str) -> List[int]:
    # (a-b],(c-d]
    print(str)
    parts = string.split(',')
    res = []
    for p in parts:
        if '-' in p:
            interval = p.split('-')
            res.extend(range(int(interval[0])+1, int(interval[1])+1))
        else:
            res.append(p)
    return res

def average_checkpoints( 
    model: Module,
    models_path: str,
    epochs: str
) -> Module:
    epochs = parse_epochs(epochs)
    print(f"average model from {epochs}")
    states_dict_list = []
    for e in epochs:
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(
            models_path + f"{e}", map_location='cpu')
        copy_model.load_state_dict(checkpoint['model'])
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list)
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)