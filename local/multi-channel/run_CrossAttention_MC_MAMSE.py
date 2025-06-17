# -*- coding: utf-8 -*-

import os
import sys
from train_Pretrain_SE import Train
from model_mc import CrossAttention_MC_MAMSE
from config import configs_mc_ca_4Speakers as train_config
import torch
from loss_function import SoftCrossEntropy_SingleTargets

from reader_mc import collate_fn_mc_mask
from reader_mc import Fbank_Embedding_Label_Mask_MC, RTTM_to_Speaker_Mask

# data="chime7_dev_array"
# feature_scp = f"data/{data}/cmn_slide_fbank_htk.list"
# oracle_rttm = f"data/{data}/oracle.rttm"
# ivector_path = "exp/nnet3_recipe_ivector/ivectors_chime7_dev_array_Oracle/ivectors_spk.txt"

# data="mixer6_train_array"
# feature_scp = f"data/{data}/cmn_slide_fbank_htk.list"
# ivector_path = f"data/{data}/ivectors_spk.txt"
# oracle_rttm = f"data/{data}/oracle.rttm"

data=
feature_scp = f"data/{data}/cmn_slide_fbank_htk.list.221"
ivector_path = f"data/{data}/ivectors_spk.txt"
oracle_rttm = f"data/{data}/oracle.rttm"

max_utt_durance = 800
batchsize = 20
mixup_rate = 0.5
num_channel = 4

label_2classes = RTTM_to_Speaker_Mask(oracle_rttm, differ_silence_inference_speech = False)

multiple_2speakers_2classes = Fbank_Embedding_Label_Mask_MC(feature_scp, ivector_path, label_2classes, append_speaker=True, min_speaker=2, max_speaker=4, max_utt_durance=max_utt_durance, frame_shift=int(max_utt_durance/4*3), num_channel=[num_channel], mixup_rate=mixup_rate, alpha=0.5)

output_dir = f"exp/CrossAttention_MC_MAMSE/Batchsize{batchsize}_4speakers_{num_channel}Channels_Segment{max_utt_durance}s_Mixup{mixup_rate}_{data}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
os.system("cp {} {}/{}".format(os.path.abspath(sys.argv[0]), output_dir, os.path.basename(sys.argv[0])))
optimizer = torch.optim.Adam
# pdb.set_trace()
train = Train(multiple_2speakers_2classes, collate_fn_mc_mask, CrossAttention_MC_MAMSE, train_config, "CrossAttention_MC_MAMSE", output_dir, optimizer, SoftCrossEntropy_SingleTargets, batchsize=batchsize, accumulation_steps=[(0, 1)], lr=0.0001, start_epoch=0, end_epoch=30, cuda=[0, 1, 2, 3], num_workers=8)
train.train(updata_utt=True)
