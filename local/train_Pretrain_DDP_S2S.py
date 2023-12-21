# -*- coding: utf-8 -*-

import time
import torch
from torch.utils import data
import utils_s2s as utils
from tqdm import tqdm
#import prefetch_generator
#import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils_s2s import get_linear_schedule_with_warmup

def save_checkpoint(local_rank, ddp_model, path):
    if local_rank== 0:
        state = {
            'model': ddp_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = Net()
    model.load_state_dict(checkpoint['model'])
    model = DDP(model, device_ids=[gpu])
    return model


class Train():
    def __init__(self, data_sets, collate_fn, model_func, model_configs, model_name, output_dir, optimizer, criterion, scheduler=None, warmup_rate=None, warmup_steps=None, checkpoint=None, batchsize=16, accumulation_steps=1, clip=-1, lr=0.001, lambda_embedding_loss=0, start_epoch=0, end_epoch=10, save_model=True, print_info=True, fix_MAMSE=False, cuda=[0], num_workers=16):
        #self.configs = model_configs
        self.output_speaker = model_configs["output_speaker"]
        self.data_sets = data_sets
        self.nnet = model_func(model_configs)
        self.optimizer = optimizer(self.nnet.parameters(), lr=lr)
        self.criterion = criterion
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.save_model = save_model
        self.print_info = print_info
        self.model_name = model_name
        self.accumulation_steps = accumulation_steps
        self.batchsize = batchsize
        self.clip = clip
        self.lambda_embedding_loss = lambda_embedding_loss
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        self.local_rank = local_rank

        self.nnet.to(device)
        if start_epoch > 0 and torch.distributed.get_rank() == 0:
            utils.load_checkpoint(self.nnet, self.optimizer, "{}/{}.model{}".format(output_dir, model_name, start_epoch))
        if checkpoint != None and os.path.isfile(checkpoint) and torch.distributed.get_rank() == 0:
            utils.load_checkpoint(self.nnet, self.optimizer, checkpoint)
        self.optimizer = optimizer(self.nnet.parameters(), lr=lr)
        self.cuda = True
        # self.nnet = torch.nn.DataParallel(self.nnet, device_ids = cuda).cuda()

        if torch.cuda.device_count() >= 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.nnet = DDP(self.nnet, device_ids=[local_rank], output_device=local_rank)
        # torch.cuda.set_device(cuda[0])
        # self.nnet = self.nnet.cuda()
        self.data_loader = data.DataLoader(data_sets, batch_size=batchsize, sampler=DistributedSampler(data_sets), collate_fn=collate_fn, drop_last=True, num_workers=num_workers)

        num_training_steps = end_epoch * len(self.data_loader)
        self.scheduler=None
        if scheduler:
            if warmup_rate:
                num_warmup_steps = num_training_steps * warmup_rate
            elif warmup_steps:
                num_warmup_steps = warmup_steps
            else:
                raise "scheduler error"
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
       
        self.output_dir = output_dir
        if torch.distributed.get_rank() == 0:
            self.logger = utils.get_logger(self.output_dir+"/log")
            self.logger.info("model_configs: {}".format(model_configs))
            self.logger.info("*" * 50)
            if self.scheduler:
                if warmup_rate:
                    self.logger.info("Using Linear warmup with rate/steps {}".format(warmup_rate))
                else:
                    self.logger.info("Using Linear warmup with rate/steps {}".format(warmup_steps))

    def train(self, updata_utt=False):
        if self.lambda_embedding_loss > 0:
            MSELoss = torch.nn.MSELoss()
        for t in range(self.start_epoch, self.end_epoch):
            try:
                accumulation_steps = self.accumulation_steps[0][1] 
                for i in self.accumulation_steps:
                    if t >= i[0]: accumulation_steps = i[1]
            except:
                accumulation_steps = self.accumulation_steps
            if torch.distributed.get_rank() == 0: 
                start_time = time.time()
                self.logger.info("Number of epoch: {}/{}, batchsize = {}".format(t + 1, self.end_epoch, accumulation_steps * self.batchsize))
            running_loss = 0.0
            chunk_loss = 0.0
            self.nnet.train()
            if t > 0 and updata_utt:
                self.data_loader.dataset.get_feature_info(random_start=True)
            if self.print_info:
                pbar = tqdm(enumerate(self.data_loader), ncols=100, total=len(self.data_loader))
            else:
                pbar = enumerate(self.data_loader)

            for i, batch_data in pbar:
                mfcc_feature, speaker_embedding, mask_data, label_data, nframes = batch_data
                if self.cuda:
                    if mfcc_feature.device.type == 'cpu': mfcc_feature = mfcc_feature.cuda()
                    if speaker_embedding.device.type == 'cpu': speaker_embedding = speaker_embedding.cuda()
                    if mask_data.device.type == 'cpu': mask_data = mask_data.cuda()
                    if label_data.device.type == 'cpu': label_data = label_data.cuda()
                ypreds = self.nnet(mfcc_feature, speaker_embedding, mask_data)
                
                loss = self.criterion(ypreds, label_data) / accumulation_steps
                
                loss.backward()
                if torch.distributed.get_rank() == 0:
                    chunk_loss += loss.item()
                    running_loss += loss.item()
                #torch.cuda.empty_cache()
                if ((i + 1) % accumulation_steps)==0:
                    if self.clip != -1:
                        torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), self.clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                        
                if ((i + 1) % (100 * accumulation_steps) == 0) and torch.distributed.get_rank() == 0:
                    self.logger.info("chunk_loss: {} {}".format((i + 1) // accumulation_steps, chunk_loss/100))
                    chunk_loss = 0
                if ((i + 1) % (1000 * accumulation_steps) == 0) and self.save_model and torch.distributed.get_rank() == 0:
                    utils.save_checkpoint(self.nnet.module, self.optimizer, "{}/{}.model{}_{}".format(self.output_dir, self.model_name, t, (i + 1) // accumulation_steps))
            
            # Save model in each iteration.
            if self.save_model and torch.distributed.get_rank() == 0:
                self.logger.info("Number of epoch: {}/{}, training set CE loss = {}".format(t + 1, self.end_epoch, running_loss/(i // accumulation_steps)))
                utils.save_checkpoint(self.nnet.module, self.optimizer, "{}/{}.model{}".format(self.output_dir, self.model_name, t + 1))
                self.logger.info("Model saved at {}/{}.model{}".format(self.output_dir, self.model_name, t + 1))
                end_time = time.time()
                self.logger.info("Time used for each eopch training: {} seconds.".format(end_time - start_time))
                self.logger.info("*" * 50)
