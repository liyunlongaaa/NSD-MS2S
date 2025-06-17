# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import math
import scipy.io as sio
import json
import copy
import HTK
import pdb
import random
import re

class LoadIVector():
    def __init__(self, speaker_embedding_txt):
        self.speaker_embedding, self.realspeaker_embedding = self.load_ivector(speaker_embedding_txt)
        self.speakers = list(self.realspeaker_embedding.keys())

    def load_ivector(self, speaker_embedding_txt):
        SCP_IO = open(speaker_embedding_txt)
        speaker_embedding = {}
        realspeaker_embedding = {}
        raw_lines = [l for l in SCP_IO]
        SCP_IO.close()
        for i in range(len(raw_lines) // 2):
            speaker = raw_lines[2*i].split()[0]
            ivector = np.array(raw_lines[2*i+1].split()[:-1], np.float32)
            speaker_embedding[speaker] = torch.from_numpy(ivector)
            real_spk = speaker.split("-")[1]
            if real_spk not in realspeaker_embedding.keys():
                realspeaker_embedding[real_spk] = []
            realspeaker_embedding[real_spk].append(torch.from_numpy(ivector))
        return speaker_embedding, realspeaker_embedding

    def get_speaker_embedding(self, speaker):
        if not speaker in self.speaker_embedding.keys():
            print("{} not in sepaker embedding list".format(speaker))
            # pdb.set_trace()
            exit()
        return self.speaker_embedding[speaker]

class LoadXVector():
    def __init__(self, speaker_embedding_txt):
        self.speaker_embedding = self.load_xvector(speaker_embedding_txt)

    def load_xvector(self, speaker_embedding_txt):
        speaker_embedding = {}
        for l in open(speaker_embedding_txt):
            speaker = os.path.basename(l).split('.')[0]
            xvector = np.loadtxt(l.rstrip())
            speaker_embedding[speaker] = torch.from_numpy(xvector.astype(np.float32))
        return speaker_embedding

    def get_speaker_embedding(self, speaker):
        #if not speaker in self.speaker_embedding.keys():
        #    print("{} not in sepaker embedding list".format(speaker))
            # pdb.set_trace()
            # exit()
        return self.speaker_embedding[speaker]

def collate_fn_mc_mask(batch):
    '''
    batch: B * (data, embedding, label)
    data: [num_channel, T, F]
    speaker_embedding: [num_channel, num_speaker, embedding_dim]
    mask_label: [num_speaker, T, C]
    '''
    num_speaker = batch[0][1].shape[1]
    length = [item[2].shape[1] for item in batch]
    ordered_index = sorted(range(len(length)), key=lambda k: length[k], reverse = True)
    #print(ordered_index)
    nframes = []
    input_data = []
    speaker_embedding = []
    label_data = []
    speaker_index = np.array(range(num_speaker))
    Num_Channel, Time, Freq = batch[ordered_index[0]][0].shape
    batch_size = len(length)
    input_data = np.zeros([batch_size, Num_Channel, Time, Freq]).astype(np.float32)
    mask_data = np.zeros([batch_size, num_speaker, Time]).astype(np.float32)
    for i, id in enumerate(ordered_index):
        np.random.shuffle(speaker_index)
        input_data[i, :, :length[id], :] = batch[id][0]
        speaker_embedding.append(batch[id][1][:, speaker_index, :])
        mask = copy.deepcopy(batch[id][2][speaker_index][..., 1])
        overlap = np.sum(mask>0, axis=0)
        mask[:, overlap>1] = 0
        mask_data[i, :, :length[id]] = mask
        #print(batch[id][2].shape)
        label_data.append(torch.from_numpy(batch[id][2][speaker_index].astype(np.float32)))
        nframes.append(length[id])
    input_data = torch.from_numpy(input_data).transpose(2, 3) # B * C * T * F => B * C * F * T
    speaker_embedding = torch.stack(speaker_embedding) # B * C * Speaker * Embedding_dim
    mask_data = torch.from_numpy(mask_data) # B * nspeaker * T
    label_data = torch.cat(label_data, dim=1)  # nspeaker * (Time_Batch1 + Time_Batch2 + ... + Time_BatchN)
    #print(torch.sum(label_data))
    return input_data, speaker_embedding, mask_data, label_data, nframes

class Fbank_Embedding_Label_Mask_MC():
    def __init__(self, feature_scp, speaker_embedding_txt, label, embedding_type='ivector', append_speaker=False, diff_speaker=False, min_speaker=0, max_speaker=2, max_utt_durance=800, frame_shift=None, num_channel=[4], mixup_rate=0, alpha=0.5):
        self.max_utt_durance = max_utt_durance
        if frame_shift == None:
            self.frame_shift = self.max_utt_durance // 2
        else:
            self.frame_shift = frame_shift
        self.num_channel = num_channel[0]
        self.label = label
        self.append_speaker = append_speaker
        self.min_speaker = min_speaker
        self.max_speaker = max_speaker
        self.diff_speaker = diff_speaker
        self.mixup_rate = mixup_rate #mixup_rate<0 means not perform mixup strategy when training
        self.alpha = alpha
        self.feature_scp = feature_scp
        self.total_frame = {}
        self.session2channle = {}
        self.feature_list = self.get_feature_info()
        # pdb.set_trace()
        self.session_to_feature_list = self.session_to_feature(self.feature_list)
        # pdb.set_trace()
        if embedding_type == 'ivector':
            self.speaker_embedding = LoadIVector(speaker_embedding_txt)
        else:
            self.speaker_embedding = LoadXVector(speaker_embedding_txt)

    def get_feature_info(self, random_start=False):
        feature_list = []
        if self.session2channle == {}:
            session2channle = {}
            with open(self.feature_scp) as SCP_IO:
                for l in SCP_IO:
                    session = os.path.basename(l).rstrip().replace('.fea', '')
                    real_session = re.sub("_U.*", "", session)
                    real_session = re.sub("_CH.*", "", real_session)
                    if real_session not in session2channle.keys():
                        session2channle[real_session] = [ (session, l.rstrip()) ]
                    else:
                        session2channle[real_session].append((session, l.rstrip()))
            self.session2channle = session2channle
        else:
            session2channle = self.session2channle
        for session in session2channle.keys():
            if session == "20090728_150523_LDC_120218" and self.num_channel >= 10:
                continue
            if(self.label.mixture_num_speaker(session) < self.min_speaker):
                continue
            if session not in self.total_frame.keys():
                num_frames = [ HTK.readHtk_info(ch[1])[0] for ch in session2channle[session] ]
                num_frames.append(self.label.get_session_length(session))
                self.total_frame[session] = min(num_frames)
            total_frame = self.total_frame[session]
            if random_start:
                cur_frame = np.random.randint(0, self.max_utt_durance-self.frame_shift)
            else:
                cur_frame = 0
            if  cur_frame + self.max_utt_durance > total_frame: continue
            while(cur_frame < total_frame):
                if cur_frame + self.max_utt_durance <= total_frame:
                    np.random.shuffle(session2channle[session])
                    cur_ch = 0
                    while cur_ch + self.num_channel <= len(session2channle[session]):
                        feature_list.append((session2channle[session][cur_ch:(cur_ch+self.num_channel)], session, cur_frame, cur_frame+self.max_utt_durance))
                        cur_ch += self.num_channel
                    cur_frame += self.frame_shift
                else:
                    cur_frame = max(0, total_frame-self.max_utt_durance)
                    np.random.shuffle(session2channle[session])
                    cur_ch = 0
                    while cur_ch + self.num_channel <= len(session2channle[session]):
                        feature_list.append((session2channle[session][cur_ch:(cur_ch+self.num_channel)], session, cur_frame, total_frame))
                        cur_ch += self.num_channel
                    break
        return feature_list

    def session_to_feature(self, feature_list):
        session_to_feature_list = {}
        for l in feature_list:
            session = l[1]
            if session not in session_to_feature_list.keys():
                session_to_feature_list[session] = []
            session_to_feature_list[session].append(l)
        return session_to_feature_list

    def load_fea(self, path, start, end):
        try:
            nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk_start_end(path, start, end)
        except:
            print("{} {} {}".format(path, start, end))
        htkdata= np.array(data).reshape(end - start, int(sampSize / 4))
        return end - start, htkdata

    def __len__(self):
        return len(self.feature_list)
    
    def get_speaker(self, speakers, num_spk=1, num_ch=1):
        np.random.shuffle(self.speaker_embedding.speakers)
        tmp_num_spk = 0
        id = 0
        speaker_embedding = []
        while tmp_num_spk < num_spk:
            is_out = True
            for spk in speakers:
                if self.speaker_embedding.speakers[id] == spk:
                    is_out = False
                    break
            if is_out:
                spk = self.speaker_embedding.speakers[id]
                try:
                    speaker_embedding += random.sample(self.speaker_embedding.realspeaker_embedding[spk], num_ch)
                except:
                    se = []
                    while len(se) < num_ch:
                        se += self.speaker_embedding.realspeaker_embedding[spk]
                    se = se[:num_ch]
                    speaker_embedding += se
                tmp_num_spk += 1
            id += 1
        return speaker_embedding

    def __getitem__(self, idx):
        l = self.feature_list[idx]
        # num_channel = np.random.choice(self.num_channel)
        session2channle, session, start, end = l
        # num_channel = min(np.random.choice(self.num_channel), len(session2channle))
        # np.random.shuffle(session2channle)
        # session2channle = session2channle[:num_channel]
        # load feature (T * F)
        data = []
        for ch in session2channle:
            data.append(self.load_fea(ch[1], start, end)[1])
        data = np.stack(data)
        # load label (Speaker * T * 3)
        mask_label, speakers = self.label.get_mixture_utternce_label(session, self.speaker_embedding, start=start, end=end, check=False)
        # pdb.set_trace()
        # load embedding (Speaker * Embedding_dim)
        # pdb.set_trace()
        if np.random.uniform() <= self.mixup_rate:
            #print(len(self.session_to_feature_list[session]))
            _, session, start, end = self.session_to_feature_list[session][np.random.choice(range(len(self.session_to_feature_list[session])))]
            data_2 = []
            for ch in session2channle:
                data_2.append(self.load_fea(ch[1], start, end)[1])
            data_2 = np.stack(data_2)
            mask_label_2, speakers_2 = self.label.get_mixture_utternce_label(session, self.speaker_embedding, start=start, end=end, check=False)
            if speakers != speakers_2:
                print("not in a same session")
                exit()
            weight = np.random.beta(self.alpha, self.alpha)
            data = weight * data + (1 - weight) * data_2
            mask_label = weight * mask_label + (1 - weight) * mask_label_2
        speaker_embedding = []
        for speaker in speakers:
            cur_spk = []
            for ch in session2channle:
                sess_spk = "{}-{}".format(ch[0], speaker)
                if sess_spk in self.speaker_embedding.speaker_embedding.keys():
                    cur_spk.append(self.speaker_embedding.get_speaker_embedding("{}-{}".format(ch[0], speaker)))
                else:
                    cur_spk = self.get_speaker(speakers, num_ch=self.num_channel)
                    break
            speaker_embedding += cur_spk

        # pdb.set_trace()
        num_speaker, T, C = mask_label.shape
        #print(mask_label.shape)
        if self.append_speaker and (num_speaker < self.max_speaker):
            append_label = np.zeros([self.max_speaker - num_speaker, T, C])
            append_label[:, :, 0] = 1
            mask_label = np.vstack([mask_label, append_label])
            speaker_embedding += self.get_speaker(speakers, num_spk=self.max_speaker-num_speaker, num_ch=self.num_channel)
        speaker_embedding = torch.stack(speaker_embedding).reshape(self.max_speaker, self.num_channel, -1).transpose(0, 1)
        if num_speaker > self.max_speaker:
            speaker_index = np.array(range(num_speaker))
            np.random.shuffle(speaker_index)
            speaker_embedding = speaker_embedding[:, speaker_index, ...][:, :self.max_speaker, ...]
            mask_label = mask_label[speaker_index][:self.max_speaker]
        '''
        returns:
        data: [num_channel, T, F]
        speaker_embedding: [num_channel, num_speaker, embedding_dim]
        mask_label: [num_speaker, T, C]
        '''
        return data, speaker_embedding, mask_label

class RTTM_to_Speaker_Mask():
    def __init__(self, oracle_rttm, differ_silence_inference_speech=False, max_speaker=8):
        self.differ_silence_inference_speech = differ_silence_inference_speech
        self.frame_label = self.get_label(oracle_rttm)
        self.max_speaker = max_speaker

    def get_label(self, oracle_rttm):
        '''
        SPEAKER session0_CH0_0L 1  116.38    3.02 <NA> <NA> 5683 <NA>
        '''
        files = open(oracle_rttm)
        MAX_len = {}
        rttm = {}
        self.all_speaker_list = []
        for line in files:
            line = line.split(" ")
            session = line[1]
            if not session in MAX_len.keys():
                MAX_len[session] = 0
            start = int(float(line[3]) * 100)
            end = int(float(line[4]) * 100) + start
            if end > MAX_len[session]:
                MAX_len[session] = end
        files.close()
        files = open(oracle_rttm)
        for line in files:
            line = line.split(" ")
            session = line[1]
            spk = line[-3]
            self.all_speaker_list.append(spk)
            if not session in rttm.keys():
                rttm[session] = {}
            if not spk in rttm[session].keys():
                if self.differ_silence_inference_speech:
                    rttm[session][spk] = np.zeros([MAX_len[session], 3], dtype=np.int8)
                else:
                    rttm[session][spk] = np.zeros([MAX_len[session], 2], dtype=np.int8)
            #print(line[3])
            start = int(float(line[3]) * 100)
            end = int(float(line[4]) * 100) + start
            rttm[session][spk][start: end, 1] = 1
        for session in rttm.keys():
            for spk in rttm[session].keys():
                rttm[session][spk][:, 0] = 1 - rttm[session][spk][:, 1]
        if self.differ_silence_inference_speech:
            for session in rttm.keys():
                num_speaker = 0
                temp_label = {}
                for spk in rttm[session].keys():
                    num_speaker += rttm[session][spk][:, 1] # sum the second dim
                for spk in rttm[session]:
                    num_inference_speaker = num_speaker - rttm[session][spk][:, 1] # get the number of no-target_speaker
                    temp_label[spk] = copy.deepcopy(rttm[session][spk])
                    without_target_speaker_mask = rttm[session][spk][:, 1] == 0 # get true when this is no-target_speaker
                    # 3 class: silence(0), target speech(1), inference speech(2)
                    temp_label[spk][without_target_speaker_mask & (num_inference_speaker>0), 0] = 0 # there is no-target_speaker
                    temp_label[spk][without_target_speaker_mask & (num_inference_speaker>0), 2] = 1
                rttm[session] = temp_label
        self.all_speaker_list = list(set(self.all_speaker_list))
        files.close()
        # pdb.set_trace()
        return rttm
            
    def mixture_num_speaker(self, session):
        return len(self.frame_label[session])

    def get_session_length(self, session):
        for spk in self.frame_label[session].keys():
            return len(self.frame_label[session][spk])

    def get_mixture_utternce_label(self, session, speaker_embedding, start=0, end=None, check=True):
        speakers = []
        mixture_utternce_label = []
        speaker_duration = []
        speaker = []
        for spk in sorted(self.frame_label[session].keys()):
            speaker_duration.append(np.sum(self.frame_label[session][spk][:, 1])) # speaker order 1,2,3,4
            speaker.append(spk)
        speaker_duration_id_order = sorted(list(range(len(speaker_duration))), reverse=True, key=lambda k:speaker_duration[k])
        cur_num_speaker = 0
        for spk_idx in speaker_duration_id_order:
            spk = speaker[spk_idx]
            if check:
                try:
                    if "{}-{}".format(session, spk) not in speaker_embedding.speaker_embedding.keys():
                        #print("{}-{}".format(session, spk))
                        continue
                except:
                    if spk not in speaker_embedding:
                        #print("{}-{}".format(session, spk))
                        continue
            if end > len(self.frame_label[session][spk]):
                print("{}-{}: {}/{}".format(session, spk, end, len(self.frame_label[session][spk])))
            mixture_utternce_label.append(self.frame_label[session][spk][start:end, :])
            speakers.append(spk)
            cur_num_speaker += 1
            if cur_num_speaker >= self.max_speaker:
                break
        return np.vstack(mixture_utternce_label).reshape(len(speakers), end - start, -1)[sorted(range(len(speakers)), key=lambda k:speakers[k])], sorted(speakers)
    
    def get_mixture_utternce_label_informed_speaker(self, session, speaker_list, start=0, end=None, max_speaker=4):
        mixture_utternce_label = []
        for spk in speaker_list:
            try:
                mixture_utternce_label.append(self.frame_label[session][spk][start:end, :])
            except:
                try:
                    real_session = session.split("_")[0]
                    mixture_utternce_label.append(self.frame_label[real_session][spk][start:end, :])
                except:
                    real_session = "_".join(session.split("_")[:-1])
                    mixture_utternce_label.append(self.frame_label[real_session][spk][start:end, :])
        mask_label = np.stack(mixture_utternce_label)
        num_speaker, T, C = mask_label.shape
        if num_speaker < max_speaker:
            append_label = np.zeros([max_speaker - num_speaker, T, C])
            append_label[:, :, 0] = 1
            mask_label = np.vstack([mask_label, append_label])
        return mask_label

    def get_mixture_utternce_label_single_speaker(self, session, target_speaker=None, start=0, end=None):
        target_speaker = target_speaker.split('_')[-1]
        if target_speaker != None:
            return self.frame_label[session][target_speaker][start: end, :]
