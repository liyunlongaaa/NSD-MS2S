# -*- coding: utf-8 -*-

from cProfile import label
import os
import numpy as np
import torch
import tqdm
import argparse

import HTK
import config
import utils
from model_mc import *
from reader import RTTM_to_Speaker_Mask
import re


def load_ivector(speaker_embedding_txt):
    SCP_IO = open(speaker_embedding_txt)
    speaker_embedding = {}
    raw_lines = [l for l in SCP_IO]
    SCP_IO.close()
    speaker_embedding_list = []
    for i in range(len(raw_lines) // 2):
        speaker = raw_lines[2*i].split()[0]
        session = "-".join(speaker.split("-")[:-1])
        real_speaker = speaker.split("-")[-1]
        if session not in speaker_embedding.keys():
            speaker_embedding[session] = {}
        ivector = torch.from_numpy(np.array(raw_lines[2*i+1].split()[:-1], np.float32))
        speaker_embedding[session][real_speaker] = ivector
        speaker_embedding_list.append(ivector)
    return speaker_embedding, speaker_embedding_list


def load_htk(path):
    nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk(path)
    htkdata = np.array(data).reshape(nSamples, int(sampSize / 4))
    #print(nSamples)
    return nSamples, htkdata

def load_multi_channel_feature(file_path, session, CHANNELs, window_len=800, hop_len=400):
    nSamples = []
    htkdata = []
    for ch in CHANNELs:
        ns, data = load_htk(file_path[f"{session}_{ch}"])
        nSamples.append(ns)
        htkdata.append(data)
    total_frame = min(nSamples)
    htkdata = np.stack([ data[:total_frame] for data in htkdata ])
    # htkdata: C * T * F
    cur_frame, feature, intervals = 0, [], []
    while(cur_frame < total_frame):
        if cur_frame + window_len <= total_frame:
            feature.append(htkdata[:, cur_frame:cur_frame+window_len, : ])
            intervals.append((cur_frame, cur_frame+window_len))
            cur_frame += hop_len
        else:
            start = max(0, total_frame-window_len)
            feature.append(htkdata[:, start:total_frame, : ])
            intervals.append((start, total_frame))
            cur_frame += window_len
    return feature, intervals, total_frame


def preds_to_rttm(preds, intervals, dur, output_path):
    rttm = np.zeros([preds[0].shape[0], dur, preds[0].shape[2]])
    weight = np.zeros(dur)
    for i, p in enumerate(preds):
        rttm[:, intervals[i][0]: intervals[i][1], :] += p
        weight[ intervals[i][0]: intervals[i][1] ] += 1
    np.save(output_path, rttm / (weight[None, :, None] + np.finfo(float).eps))


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #torch.cuda.set_device(0)
    model_name = os.path.basename(args.model_path).split(".")[0]
    nnet = eval(model_name)(config.configs[args.model_config])
    utils.load_checkpoint(nnet, None, args.model_path)
    gpus = list(range(len(args.gpu.split(','))))
    nnet = torch.nn.DataParallel(nnet, device_ids = gpus).cuda()
    #nnet = nnet.cuda()
    nnet.eval()
    softmax = torch.nn.Softmax(dim=2)
    if args.set.find("chime") != -1:
        CHANNELs = ["U01_CH1", "U02_CH1", "U03_CH1", "U04_CH1", "U06_CH1"]
        #CHANNELs = ["U01_CH1", "U01_CH4", "U02_CH1", "U02_CH4", "U03_CH1", "U03_CH4", "U04_CH1", "U04_CH4","U06_CH1", "U06_CH4"]
    elif args.set.find("dipco") != -1:
        CHANNELs = ["U01_CH1", "U02_CH1", "U03_CH1", "U04_CH1", "U05_CH1"]
    elif args.set.find("mixer6") != -1:
        CHANNELs = [ "CH{:02d}".format(c) for c in range(4, 14) ]
    #CHANNELs = ["U01_CH1", "U01_CH4", "U02_CH1", "U02_CH4", "U03_CH1", "U03_CH4", "U04_CH1", "U04_CH4","U06_CH1", "U06_CH4"]
    #CHANNELs = ["U06_CH1"]
    #CHANNELs = [ f"U0{u}_CH{c}" for u in [1,2,3,4,6] for c in range(1, 5) ]
    # CHANNELs = [ "CH04", "CH06", "CH08", "CH10", "CH11" ]
    sessions = []
    file_list = {}
    with open(args.feature_list) as INPUT:
        for l in INPUT:
            session = os.path.basename(l).split('.')[0]
            real_session = re.sub("_U.*", "", session)
            real_session = re.sub("_CH.*", "", real_session)
            if real_session not in sessions:
                sessions.append(real_session)
            file_list[session] = l.rstrip()

    label_init = RTTM_to_Speaker_Mask(args.init_rttm)
    embedding, _ = load_ivector(args.embedding_list)
    _, train_set_speaker_embedding = load_ivector(args.train_set_speaker_embedding_list)
    idxs = list(range(len(train_set_speaker_embedding)))
    for session in tqdm.tqdm(sessions, ncols=100):
        #print(f"processing {session}")
        # if session == "S09":
        #     CHANNELs = [ f"U0{u}_CH{c}" for u in [1,2,3,4,6] for c in range(1, 3) ]
        # else:
        #     CHANNELs = [ f"U0{u}_CH{c}" for u in [1,2,3,4,5,6] for c in range(1, 3) ]
        output_path = os.path.join(args.output_dir, session+".npy")
        if os.path.isfile(output_path):
            continue
        speaker_embedding = []
        # if session == "20090728_150523_LDC_120218":
        #     CHANNELs = [ "CH{:02d}".format(c) for c in range(1, 14) if c != 8 ]
        # else:
        #     CHANNELs = [ "CH{:02d}".format(c) for c in range(4, 14) ]
        speaker_list = list(embedding[f"{session}_{CHANNELs[0]}"].keys())
        num_speaker = len(speaker_list)
        for ch in CHANNELs:
            for spk in speaker_list:
                speaker_embedding.append(embedding[f"{session}_{ch}"][spk])
        for idx in np.random.choice(idxs, (args.max_speaker - num_speaker)*args.num_channel, replace=False):
            speaker_embedding.append(train_set_speaker_embedding[idx])
        speaker_embedding = torch.stack(speaker_embedding).reshape(args.num_channel, args.max_speaker, -1) # num_speaker * embedding_dim
        feature, intervals, total_frame = load_multi_channel_feature(file_list, session, CHANNELs, args.max_utt_durance, args.hop_len)
        output_path = os.path.join(args.output_dir, session)
        preds, i, cur_utt, batch, batch_intervals, new_intervals = [], 0, 0, [], [], []
        with torch.no_grad():
            for m in feature:
                batch.append(torch.from_numpy(m.astype(np.float32)))
                batch_intervals.append(intervals[cur_utt])
                cur_utt += 1
                i += 1
                if (i == args.batch_size) or (len(feature) == cur_utt):
                    length = [item.shape[1] for item in batch]
                    ordered_index = sorted(range(len(length)), key=lambda k: length[k], reverse = True)
                    C, Time, Freq = batch[ordered_index[0]].shape
                    cur_batch_size = len(length)
                    input_data = np.zeros([cur_batch_size, C, Time, Freq]).astype(np.float32)
                    mask_data = np.zeros([cur_batch_size, args.max_speaker, Time]).astype(np.float32)
                    nframes = []
                    batch_speaker_embedding = []
                    for i, id in enumerate(ordered_index):
                        input_data[i, :, :length[id], :] = batch[id]
                        nframes.append(length[id])
                        batch_speaker_embedding.append(speaker_embedding)
                        mask = label_init.get_mixture_utternce_label_informed_speaker(session, speaker_list, start=batch_intervals[id][0], end=batch_intervals[id][1], max_speaker=args.max_speaker)[..., 1]
                        if args.remove_overlap:
                            overlap = np.sum(mask, axis=0)
                            mask[:, overlap>=2] = 0
                        mask_data[i, :, :mask.shape[1]] = mask
                        new_intervals.append(batch_intervals[id])
                    input_data = torch.from_numpy(input_data).transpose(2, 3).cuda()
                    batch_speaker_embedding = torch.stack(batch_speaker_embedding).cuda() # B * C * num_speaker * embedding_dim
                    mask_data = torch.from_numpy(mask_data).cuda()
                    ypreds = nnet(input_data, batch_speaker_embedding, mask_data, torch.from_numpy(np.array(nframes, dtype=np.int32)))
                    ypreds = torch.stack([k for k in ypreds]) # speaker * T * 3
                    ypreds = softmax(ypreds).detach().cpu().numpy()
                    cur_frame = 0
                    for n in nframes:
                        #print(n)
                        preds.append(ypreds[:num_speaker, cur_frame:(cur_frame+n), :])
                        cur_frame += n
                    i, batch, batch_intervals = 0, [], []
        preds_to_rttm(preds, new_intervals, total_frame, output_path)

def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')
    parser.add_argument('--embedding_list', metavar='PATH', required=True,
                        help='embedding_list.')
    parser.add_argument('--set', type=str, default="chime6",
                        help='data set [chime6, dipco, mixer6].')
    parser.add_argument('--gpu', type=str, default="0",
                        help='data set [chime6, dipco, mixer6].')
    parser.add_argument('--train_set_speaker_embedding_list', metavar='PATH', required=True,
                        help='train_set_speaker_embedding_list.')
    parser.add_argument('--feature_list', metavar='PATH', required=True,
                        help='feature_list')
    parser.add_argument('--model_path', metavar='PATH', required=True,
                        help='model_path.')  
    parser.add_argument('--output_dir', metavar='PATH', required=True,
                        help='output_dir.')                       
    parser.add_argument('--max_speaker', metavar='PATH', type=int, default=8,
                help='max_speaker.')
    parser.add_argument('--init_rttm', metavar='PATH', required=True,
                        help='init_rttm.')
    parser.add_argument('--model_config', metavar='PATH', type=str, default="configs_4Speakers_ivectors128_2Classes",
                help='domain_list.')
    parser.add_argument('--num_channel', metavar='PATH', type=int, default=10,
                help='num_channel.')
    parser.add_argument('--max_utt_durance', metavar='PATH', type=int, default=800*32,
                help='max_utt_durance.')
    parser.add_argument('--hop_len', metavar='PATH', type=int, default=100,
                help='hop_len.')
    parser.add_argument('--split_seg', metavar='PATH', type=int, default=800,
                help='split_seg.')
    parser.add_argument('--batch_size', metavar='PATH', type=int, default=8,
                help='batch_size.')
    parser.add_argument('--remove_overlap', action="store_true", 
                help='remove_overlap.')      
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
