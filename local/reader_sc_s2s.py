# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import math
import scipy.io as sio
import json
import copy
import HTK
import re
import tqdm


class LoadIVector():
    def __init__(self, speaker_embedding_txt):
        self.speaker_embedding = self.load_ivector(speaker_embedding_txt)

    def load_ivector(self, speaker_embedding_txt):
        SCP_IO = open(speaker_embedding_txt)
        speaker_embedding = {}
        raw_lines = [l for l in SCP_IO]
        SCP_IO.close()
        for i in tqdm.tqdm(range(len(raw_lines) // 2), ncols=100):
            speaker = raw_lines[2*i].split()[0]
            ivector = np.array(raw_lines[2*i+1].split()[:-1], np.float32)
            speaker_embedding[speaker] = torch.from_numpy(ivector)
        return speaker_embedding

    def get_speaker_embedding(self, speaker):
        if not speaker in self.speaker_embedding.keys():
            print("{} not in sepaker embedding list".format(speaker))
            # pdb.set_trace()
        return self.speaker_embedding[speaker]

def collate_fn_mask(batch):
    '''
    batch: B * (data, embedding, label)
    data: [T, F]
    speaker_embedding: [num_speaker, embedding_dim]
    mask_label: [num_speaker, T, C]
    '''
    batch = [ b for b in batch if b != None ]
    num_speaker = batch[0][1].shape[0]
    length = [item[2].shape[1] for item in batch]
    ordered_index = sorted(range(len(length)), key=lambda k: length[k], reverse = True) 
    #print(ordered_index)
    nframes = []
    input_data = []
    speaker_embedding = []
    label_data = []
    speaker_index = np.array(range(num_speaker))
    #print("speaker_index :", speaker_index)
    #print("ordered_index :", ordered_index)

    Time, Freq = batch[ordered_index[0]][0].shape  #用最长的开矩阵
    batch_size = len(length)
    input_data = np.zeros([batch_size, Time, Freq]).astype(np.float32)
    mask_data = np.zeros([batch_size, num_speaker, Time]).astype(np.float32)
    for i, id in enumerate(ordered_index):
        np.random.shuffle(speaker_index)
        #print("shuffle speaker_index", speaker_index)
        input_data[i, :length[id], :] = batch[id][0]
        speaker_embedding.append(batch[id][1][speaker_index])
        mask = copy.deepcopy(batch[id][2][speaker_index]) # (N, T)
        overlap = np.sum(mask>0, axis=0)
        mask[:, overlap>1] = 0    #0和大于1的都mask
        mask_data[i, :, :length[id]] = mask
        #print(batch[id][2].shape)
        label_data.append(torch.from_numpy(batch[id][2][speaker_index].astype(np.float32)))
        nframes.append(length[id])
    input_data = torch.from_numpy(input_data).transpose(1, 2) # B * T * F => B * F * T
    speaker_embedding = torch.stack(speaker_embedding) # B * Speaker * Embedding_dim
    mask_data = torch.from_numpy(mask_data) # B * nspeaker * T
    label_data = torch.stack(label_data)  # B * nspeaker * T
    #print(torch.sum(label_data))
    return input_data, speaker_embedding, mask_data, label_data, nframes

class Fbank_Embedding_Label_Mask():
    def __init__(self, feature_scp, speaker_embedding_txt, label=None, train_segments=None, embedding_type='ivector', append_speaker=False, diff_speaker=False, min_speaker=0, max_speaker=2, max_utt_durance=800, frame_shift=None, min_segments=0, mixup_rate=0, alpha=0.5):
        self.feature_scp = feature_scp
        self.max_utt_durance = max_utt_durance
        if frame_shift == None:
            self.frame_shift = self.max_utt_durance // 2
        else:
            self.frame_shift = frame_shift
        self.label = label
        self.append_speaker = append_speaker
        self.min_speaker = min_speaker
        self.max_speaker = max_speaker
        self.diff_speaker = diff_speaker
        self.mixup_rate = mixup_rate #mixup_rate<0 means not perform mixup strategy when training
        self.alpha = alpha
        self.feature_list, self.session_to_feapath = self.get_feature_info(train_segments, min_segments)
        # pdb.set_trace()
        self.session_to_feature_list = self.session_to_feature(self.feature_list)
        # pdb.set_trace()
        self.speaker_embedding = LoadIVector(speaker_embedding_txt)
        self.data_set_speaker = list(self.speaker_embedding.speaker_embedding.keys())


    def prepare_segment(self, session, cur_frame, cur_end):
        feature_list = []
        while(cur_frame < cur_end):
            if cur_frame + self.max_utt_durance < cur_end:
                feature_list.append((session, cur_frame, cur_frame+self.max_utt_durance))
                cur_frame += self.frame_shift
            else:
                cur_frame = max(0, cur_end-self.max_utt_durance)
                feature_list.append((session, cur_frame, cur_end))
                break
        return feature_list

    def get_feature_info(self, train_segments=None, min_segments=0, random_start=False):
        feature_list = []
        session_to_feapath = {}
        segments = {}
        if train_segments != None:
            with open(train_segments) as IN:
                for l in IN:
                    session, s ,e = l.rstrip().split()
                    if session not in segments.keys():
                        segments[session] = []
                    segments[session].append([int(s), int(e)])
        with open(self.feature_scp) as SCP_IO:
            files = [ l for l in SCP_IO ]
            for l in tqdm.tqdm(files, ncols=100):
                '''
                basename: 0000.fea
                '''
                session = os.path.basename(l).rstrip().replace('.fea', '')
                session_to_feapath[session] = l.rstrip()
                real_session = re.sub("_U.*", "", session)
                real_session = re.sub("_CH.*", "", real_session)
                # if(self.label != None and self.label.mixture_num_speaker(session) < self.min_speaker):
                #     continue
                try:
                    total_frame = HTK.readHtk_info(l.rstrip())[0]
                except:
                    print(l)
                    continue
                if self.label != None:
                    session_length = self.label.get_session_length(real_session)
                    MAX_LEN = min(total_frame, session_length)
                else:
                    MAX_LEN = total_frame
                s = [0, MAX_LEN]
                if random_start:
                    cur_frame = np.random.randint(s[0], s[0]+self.max_utt_durance-self.frame_shift)
                else:
                    cur_frame = s[0]
                cur_end = min(s[1], MAX_LEN)
                if cur_end - cur_frame < min(self.max_utt_durance, min_segments): continue
                feature_list += self.prepare_segment(session, cur_frame, cur_end)
                
        return feature_list, session_to_feapath

    def session_to_feature(self, feature_list):
        session_to_feature_list = {}
        for l in feature_list:
            session = l[0]
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
    
    def __getitem__(self, idx):
        l = self.feature_list[idx]
        session, start, end = l
        real_session = re.sub("_U.*", "", session)
        real_session = re.sub("_CH.*", "", real_session)
        path = self.session_to_feapath[session]
        # load feature (T * F)
        _, data = self.load_fea(path, start, end)
        # load label (Speaker * T * 3)
        try:
            mask_label, speakers = self.label.get_mixture_utternce_label(real_session, self.speaker_embedding, raw_session=session, start=start, end=end)
        except:
            print(l)
            return None
        # pdb.set_trace()
        # load embedding (Speaker * Embedding_dim)
        # pdb.set_trace()
        if np.random.uniform() <= self.mixup_rate:
            #print(len(self.session_to_feature_list[session]))
            session, start, end = self.session_to_feature_list[session][np.random.choice(range(len(self.session_to_feature_list[session])))]
            _, data_2 = self.load_fea(path, start, end)
            try:
                mask_label_2, speakers_2 = self.label.get_mixture_utternce_label(real_session, self.speaker_embedding, raw_session=session, start=start, end=end)
            except:
                print(l)
                return None
            if speakers != speakers_2:
                print("not in a same session")
                return None
            weight = np.random.beta(self.alpha, self.alpha)
            data = weight * data + (1 - weight) * data_2
            mask_label = weight * mask_label + (1 - weight) * mask_label_2
        speaker_embedding = []
        for speaker in speakers:
            speaker_embedding.append(self.speaker_embedding.get_speaker_embedding("{}-{}".format(session, speaker)))
        # pdb.set_trace()
        num_speaker, T = mask_label.shape
        #print(mask_label.shape)
        if self.append_speaker and (num_speaker < self.max_speaker):
            append_label = np.zeros([self.max_speaker - num_speaker, T])
            #append_label[:, :, 0] = 1
            mask_label = np.vstack([mask_label, append_label])
            for speaker in np.random.choice(self.data_set_speaker, self.max_speaker - num_speaker, replace=False):
                speaker_embedding.append(self.speaker_embedding.get_speaker_embedding(speaker))
        speaker_embedding = torch.stack(speaker_embedding)
        if num_speaker > self.max_speaker:
            speaker_index = np.array(range(num_speaker))
            np.random.shuffle(speaker_index)
            speaker_embedding = speaker_embedding[speaker_index][:self.max_speaker]
            mask_label = mask_label[speaker_index][:self.max_speaker]
        '''
        returns:
        data: [T, F]
        speaker_embedding: [num_speaker, embedding_dim]
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
        IO = open(oracle_rttm)
        files = [ l for l in IO ]
        IO.close()
        MAX_len = {}
        rttm = {}
        utts = []
        speech = np.array([1])
        for line in tqdm.tqdm(files, ncols=100):
            line = line.split(" ")
            session = line[1]
            spk = line[-3]
            if not session in MAX_len.keys():
                MAX_len[session] = 0
            start = int(float(line[3]) * 100)
            end = int(float(line[4]) * 100) + start
            if end > MAX_len[session]:
                MAX_len[session] = end
            utts.append([session, spk, start, end])
        for utt in tqdm.tqdm(utts, ncols=100):
            session, spk, start, end = utt
            if not session in rttm.keys():
                rttm[session] = {}
            if not spk in rttm[session].keys():
                rttm[session][spk] = np.zeros(MAX_len[session], dtype=np.int8)
                #rttm[session][spk][:, 0] = 1 #2 dim
            rttm[session][spk][start: end] = speech
        return rttm
            
    def mixture_num_speaker(self, session):
        #print(session)
        return len(self.frame_label[session])

    def get_session_length(self, session):
        for spk in self.frame_label[session].keys():
            return len(self.frame_label[session][spk])

    def get_mixture_utternce_label(self, session, speaker_embedding, raw_session=None, start=0, end=None, check=True):
        speakers = []
        mixture_utternce_label = []
        speaker_duration = []
        speaker = []
        if raw_session == None:
            raw_session = session
        for spk in sorted(self.frame_label[session].keys()):
            speaker_duration.append(np.sum(self.frame_label[session][spk])) # speaker order 1,2,3,4
            speaker.append(spk)
        speaker_duration_id_order = sorted(list(range(len(speaker_duration))), reverse=True, key=lambda k:speaker_duration[k])
        cur_num_speaker = 0
        for spk_idx in speaker_duration_id_order:
            spk = speaker[spk_idx]
            if check:
                try:
                    if "{}-{}".format(raw_session, spk) not in speaker_embedding.speaker_embedding.keys():
                        #print("{}-{}".format(session, spk))
                        continue
                except:
                    if spk not in speaker_embedding:
                        #print("{}-{}".format(session, spk))
                        continue
            if end > len(self.frame_label[session][spk]):
                print("{}-{}: {}/{}".format(session, spk, end, len(self.frame_label[session][spk])))
            mixture_utternce_label.append(self.frame_label[session][spk][start:end])
            speakers.append(spk)
            cur_num_speaker += 1
            if cur_num_speaker >= self.max_speaker:
                break
        return np.vstack(mixture_utternce_label).reshape(len(speakers), end - start)[sorted(range(len(speakers)), key=lambda k:speakers[k])], sorted(speakers) # (n_spk, len)
    
    def get_mixture_utternce_label_informed_speaker(self, session, speaker_list, start=0, end=None, max_speaker=4):
        mixture_utternce_label = []
        for spk in speaker_list:
            try:
                mixture_utternce_label.append(self.frame_label[session][spk][start:end])
            except:
                print(session, self.frame_label.keys())
                real_session = session.split("_")[0]
                mixture_utternce_label.append(self.frame_label[real_session][spk][start:end])
        mask_label = np.stack(mixture_utternce_label)
        num_speaker, T = mask_label.shape
        if num_speaker < max_speaker:
            append_label = np.zeros([max_speaker - num_speaker, T])
            #append_label[:, :, 0] = 1
            mask_label = np.vstack([mask_label, append_label])
        return mask_label

    def get_mixture_utternce_label_single_speaker(self, session, target_speaker=None, start=0, end=None):
        target_speaker = target_speaker.split('_')[-1]
        if target_speaker != None:
            return self.frame_label[session][target_speaker][start: end]
