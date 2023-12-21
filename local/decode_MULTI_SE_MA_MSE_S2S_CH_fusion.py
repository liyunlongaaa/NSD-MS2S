
#from cProfile import label
import os
import numpy as np
import torch
import tqdm
import argparse

import HTK
import config
import utils
import importlib
from reader_s2s import RTTM_to_Speaker_Mask


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
    return nSamples, htkdata

def load_single_channel_feature(file_path, window_len=800, hop_len=400):
    nSamples, htkdata = load_htk(file_path)
    # htkdata: T * F
    cur_frame, feature, intervals, total_frame = 0, [], [], nSamples
    while(cur_frame < total_frame):
        if cur_frame + window_len <= total_frame:
            feature.append(htkdata[cur_frame:cur_frame+window_len, : ])
            intervals.append((cur_frame, cur_frame+window_len))
            cur_frame += hop_len
        else:
            start = max(0, total_frame-window_len)
            feature.append(htkdata[start:total_frame, : ])
            intervals.append((start, total_frame))
            cur_frame += window_len
    return feature, intervals, total_frame

def preds_to_rttm(preds_intervals_list, total_dur, output_path, session_root):
    rttm = np.zeros([preds_intervals_list[0][0][0].shape[0], total_dur])
    ch_len = len(preds_intervals_list)
    for i, (preds, intervals) in  enumerate(preds_intervals_list):
        cur_rttm = np.zeros([preds_intervals_list[0][0][0].shape[0], total_dur])
        cur_weight = np.zeros(total_dur)
        for i, p in enumerate(preds):
            cur_rttm[:, intervals[i][0]: intervals[i][1]] += p
            cur_weight[ intervals[i][0]: intervals[i][1] ] += 1         
        cur_rttm = cur_rttm / (cur_weight[None, :] + np.finfo(float).eps)    
        rttm += cur_rttm
    np.save(output_path, rttm / ch_len)  #Average probability of all channels

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("model type is {}".format(args.model_type))

    model = importlib.import_module('.', package='{}'.format(args.model_type))
    try:
        if args.model_config != "N":
            print(config.configs[args.model_config])
            nnet = model.MULTI_MAM_SE_S2S_model(config.configs[args.model_config])
        else:
            configs = torch.load(args.model_path)["configs"]
            print("configs from model ", configs)
            nnet = model.MULTI_MAM_SE_S2S_model(configs)
    except:
        #nnet = SE_MA_MSE_NSD(config.configs[args.model_config])
        raise "load model error!!!"
    utils.load_checkpoint(nnet, None, args.model_path)

    gpus = list(range(len(args.gpu.split(','))))
    nnet = torch.nn.DataParallel(nnet, device_ids = gpus).cuda()
    nnet.eval()
    Sigmoid = torch.nn.Sigmoid()  

    file_list = {}
    with open(args.feature_list) as INPUT:
        for l in INPUT:
            session = os.path.basename(l).split('.')[0]
            file_list[session] = l.rstrip()

    label_init = RTTM_to_Speaker_Mask(args.init_rttm)
    embedding, _ = load_ivector(args.embedding_list)
    _, train_set_speaker_embedding = load_ivector(args.train_set_speaker_embedding_list)
    idxs = list(range(len(train_set_speaker_embedding)))

    session2CH = {}
    for session_CH in file_list.keys():
        if len(session_CH.split('_')) == 3:
            session = session_CH.split('_')[0]  # chime6 dipco
        elif len(session_CH.split('_')) == 5:
            session = session_CH.rsplit('_', 1)[0] #mixer6
        else:
            print(session_CH)
            raise "format error"
        if session not in session2CH:
            session2CH[session] = []
        session2CH[session].append(session_CH)

    session_root2muti_preds = {}

    for session_root in tqdm.tqdm(session2CH):
        session_root2muti_preds[session_root] = []
        max_total_frame = -1
        for session in session2CH[session_root]:
            speaker_embedding = []
            speaker_list = list(embedding[session].keys())
            num_speaker = len(speaker_list)
            if num_speaker > args.max_speaker: print(speaker_list)
            for spk in embedding[session].keys():
                speaker_embedding.append(embedding[session][spk])
            for idx in np.random.choice(idxs, args.max_speaker - num_speaker, replace=False):
                speaker_embedding.append(train_set_speaker_embedding[idx])
            speaker_embedding = torch.stack(speaker_embedding) # num_speaker * embedding_dim
            feature, intervals, total_frame = load_single_channel_feature(file_list[session], args.max_utt_durance, args.hop_len)
            max_total_frame = max(max_total_frame, total_frame)
            preds, i, cur_utt, batch, batch_intervals, new_intervals = [], 0, 0, [], [], []
            with torch.no_grad():
                for m in feature:
                    batch.append(torch.from_numpy(m.astype(np.float32)))
                    batch_intervals.append(intervals[cur_utt])
                    cur_utt += 1
                    i += 1
                    if (i == args.batch_size) or (len(feature) == cur_utt): #sufficient one batch or at last, begin infer
                        length = [item.shape[0] for item in batch]
                        ordered_index = sorted(range(len(length)), key=lambda k: length[k], reverse = True)
                        Time, Freq = batch[ordered_index[0]].shape
                        cur_batch_size = len(length)
                        input_data = np.zeros([cur_batch_size, Time, Freq]).astype(np.float32)
                        mask_data = np.zeros([cur_batch_size, args.max_speaker, Time]).astype(np.float32)
                        nframes = []
                        batch_speaker_embedding = []
                        for i, id in enumerate(ordered_index):
                            input_data[i, :length[id], :] = batch[id]
                            nframes.append(length[id])
                            batch_speaker_embedding.append(speaker_embedding)
                            mask = label_init.get_mixture_utternce_label_informed_speaker(session, speaker_list, start=batch_intervals[id][0], end=batch_intervals[id][1], max_speaker=args.max_speaker)
                            if args.remove_overlap: 
                                overlap = np.sum(mask, axis=0)
                                mask[:, overlap>=2] = 0
                            mask_data[i, :, :mask.shape[1]] = mask
                            new_intervals.append(batch_intervals[id])
                        input_data = torch.from_numpy(input_data).transpose(1, 2).cuda()
                        
                        batch_speaker_embedding = torch.stack(batch_speaker_embedding).cuda() # B * num_speaker * embedding_dim
                        mask_data = torch.from_numpy(mask_data).cuda()
                        ypreds = nnet(input_data, batch_speaker_embedding, mask_data)
                        if isinstance(ypreds, (list, tuple)):
                            ypreds = ypreds[0]
                        ypreds = ypreds[:, :num_speaker, :]
                        ypreds = Sigmoid(ypreds).transpose(0, 1).reshape(num_speaker, -1).detach().cpu().numpy()  #B, N, T -> N -> N, (B,T) 

                        cur_frame = 0
                        for n in nframes:
                            #print(n)
                            preds.append(ypreds[:, cur_frame:(cur_frame+n)])
                            cur_frame += n
                        i, batch, batch_intervals = 0, [], []
            session_root2muti_preds[session_root].append([preds, new_intervals])
        output_path = os.path.join(args.output_dir, session_root)
        preds_to_rttm(session_root2muti_preds[session_root], max_total_frame, output_path, session_root)

def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')
    parser.add_argument('--embedding_list', metavar='PATH', required=True,
                        help='embedding_list.')  
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
    parser.add_argument('--model_config', metavar='PATH', type=str, default="N",
                help='domain_list.')
    parser.add_argument('--max_utt_durance', metavar='PATH', type=int, default=800*32,
                help='max_utt_durance.')
    parser.add_argument('--hop_len', metavar='PATH', type=int, default=100,
                help='hop_len.')
    parser.add_argument('--split_seg', metavar='PATH', type=int, default=0,
                help='split_seg.')
    parser.add_argument('--batch_size', metavar='PATH', type=int, default=8,
                help='batch_size.')
    parser.add_argument('--gpu', type=str, default="0",
                        help='gpu')
    parser.add_argument('--remove_overlap', action="store_true", 
                help='remove_overlap.')      
    parser.add_argument('--model_type', metavar='Model_Type', type=str, default="model_S2S",
                help='model_type.')  
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)

