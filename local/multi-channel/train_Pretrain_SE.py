# -*- coding: utf-8 -*-

import time
import torch
from torch.utils import data
import utils
from tqdm import tqdm
#import prefetch_generator
#import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

class Train():
    def __init__(self, data_sets, collate_fn, model_func, model_configs, model_name, output_dir, optimizer, criterion, checkpoint=None, batchsize=16, accumulation_steps=1, clip=-1, lr=0.001, split_seg=-1, lambda_embedding_loss=0, start_epoch=0, end_epoch=10, save_model=True, print_info=True, fix_MAMSE=False, cuda=[0,1,2,3], num_workers=16):
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
        self.split_seg = split_seg
        self.lambda_embedding_loss = lambda_embedding_loss
        self.data_loader = data.DataLoader(data_sets, batch_size=batchsize, collate_fn=collate_fn, shuffle=True, drop_last=True, num_workers=num_workers)
        if start_epoch > 0:
            print("load model from {}".format("{}/{}.model{}".format(output_dir, model_name, start_epoch)))
            utils.load_checkpoint(self.nnet, self.optimizer, "{}/{}.model{}".format(output_dir, model_name, start_epoch))
        if checkpoint != None and os.path.isfile(checkpoint):
            utils.load_checkpoint(self.nnet, self.optimizer, checkpoint)
        if fix_MAMSE:
            for p in self.nnet.mamse1.parameters():
                p.requires_grad = False
            for p in self.nnet.mamse2.parameters():
                p.requires_grad = False
            self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.nnet.parameters()), lr=lr)
        else:
            self.optimizer = optimizer(self.nnet.parameters(), lr=lr)
        self.cuda = True
        self.nnet = torch.nn.DataParallel(self.nnet, device_ids = cuda).cuda()
        # torch.cuda.set_device(cuda[0])
        # self.nnet = self.nnet.cuda()
        self.output_dir = output_dir
        self.logger = utils.get_logger(self.output_dir+"/log")
        self.logger.info("model_configs: {}".format(model_configs))
        self.logger.info("*" * 50)

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
            start_time = time.time()
            self.logger.info("Number of epoch: {}/{}, batchsize = {}".format(t + 1, self.end_epoch, accumulation_steps * self.batchsize))
            running_loss = 0.0
            chunk_loss = 0.0
            self.nnet.train()
            if t > 0 and updata_utt:
                self.data_loader.dataset.get_feature_info(random_start=True)
            if self.print_info:
                pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
            else:
                pbar = enumerate(self.data_loader)
            point = time.time()

            for i, batch_data in pbar:
                #print("read time: {}".format(time.time() - point))
                #point = time.time()
                #pdb.set_trace()
                mfcc_feature, speaker_embedding, mask_data, label_data, nframes = batch_data
                if self.cuda:
                    if mfcc_feature.device.type == 'cpu': mfcc_feature = mfcc_feature.cuda()
                    if speaker_embedding.device.type == 'cpu': speaker_embedding = speaker_embedding.cuda()
                    if mask_data.device.type == 'cpu': mask_data = mask_data.cuda()
                    if label_data.device.type == 'cpu': label_data = label_data.cuda()
                #print("tocuda time: {}".format(time.time() - point))
                #point = time.time()
                
                # pdb.set_trace()
                if self.lambda_embedding_loss > 0:
                    ypreds = self.nnet(mfcc_feature, speaker_embedding, mask_data, torch.from_numpy(np.array(nframes, dtype=np.int)), split_seg=self.split_seg, return_embedding=True)
                    loss = (self.criterion(ypreds[0], label_data) + self.lambda_embedding_loss * MSELoss(ypreds[1], speaker_embedding)) / accumulation_steps
                else:
                    ypreds = self.nnet(mfcc_feature, speaker_embedding, mask_data, torch.from_numpy(np.array(nframes, dtype=np.int)), split_seg=self.split_seg)
                    loss = self.criterion(ypreds, label_data) / accumulation_steps
                #print("forward time: {}".format(time.time() - point))
                #point = time.time()
                chunk_loss += loss.item()
                running_loss += loss.item()
                loss.backward()
                #torch.cuda.empty_cache()
                if ((i + 1) % accumulation_steps)==0:
                    if self.clip != -1:
                        torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), self.clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if ((i + 1) % (100 * accumulation_steps) == 0):
                    self.logger.info("chunk_loss: {} {}".format((i + 1) // accumulation_steps, chunk_loss/100))
                    chunk_loss = 0
                if ((i + 1) % (2000 * accumulation_steps) == 0) and self.save_model:
                    utils.save_checkpoint(self.nnet.module, self.optimizer, "{}/{}.model{}_{}".format(self.output_dir, self.model_name, t, (i + 1) // accumulation_steps))
                #print("backward time: {}".format(time.time() - point))
                #point = time.time()
            self.logger.info("Number of epoch: {}/{}, training set CE loss = {}".format(t + 1, self.end_epoch, running_loss/(i // accumulation_steps)))
            # Save model in each iteration.
            if self.save_model:
                utils.save_checkpoint(self.nnet.module, self.optimizer, "{}/{}.model{}".format(self.output_dir, self.model_name, t + 1))
                self.logger.info("Model saved at {}/{}.model{}".format(self.output_dir, self.model_name, t + 1))
            end_time = time.time()
            self.logger.info("Time used for each eopch training: {} seconds.".format(end_time - start_time))
            self.logger.info("*" * 50)
