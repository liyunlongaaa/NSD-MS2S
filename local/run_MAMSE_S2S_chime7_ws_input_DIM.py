# -*- coding: utf-8 -*-

import os
import sys
from train_Pretrain_DDP_S2S import Train
from model_S2S_weight_input_DIM import MULTI_MAM_SE_S2S_model
from config import configs3_4Speakers_ivector_ivector128_xvectors128_S2S_MA_MSE_DIM as config_train
import torch
from reader_sc_s2s import Fbank_Embedding_Label_Mask, collate_fn_mask, RTTM_to_Speaker_Mask


data="CHiME6MAMSELabel_SimuCHiME6_Mixer6MAMSELabel_SimuMixer6_SimuDipcoDevNoise" # train data name
feature_scp = f"data/{data}/cmn_slide_fbank_htk.list"  # fbank
ivector_path = f"data/{data}/ivectors_spk.txt"    # i-vector
oracle_rttm = f"data/{data}/oracle.rttm"

max_utt_durance = 800
batchsize = 20
mixup_rate=0.5

output_dir = f"exp/S2S/Batchsize{batchsize}_4speakers_Segment{max_utt_durance}s_Mixup{mixup_rate}_{data}_all_data_512_all0Dropout_6layers_weight_input_DIM"
print('exp will be saved in', output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
label_2classes = RTTM_to_Speaker_Mask(oracle_rttm, differ_silence_inference_speech = False)

multiple_4speakers_2classes = Fbank_Embedding_Label_Mask(feature_scp, ivector_path, label_2classes, append_speaker=True, diff_speaker=True, min_speaker=2, max_speaker=4, max_utt_durance=max_utt_durance, frame_shift=int(max_utt_durance/4*3), mixup_rate=mixup_rate, alpha=0.5)


os.system("cp {} {}/{}".format(os.path.abspath(sys.argv[0]), output_dir, os.path.basename(sys.argv[0])))
os.system("cp {} {}/{}".format("local_gb/model_S2S_weight_input_DIM.py", output_dir, "model.py"))
optimizer = torch.optim.Adam
loss_fn = torch.nn.BCEWithLogitsLoss()


train = Train(multiple_4speakers_2classes, collate_fn_mask, MULTI_MAM_SE_S2S_model, config_train, "MULTI_MAM_SE_S2S_model", output_dir, optimizer, loss_fn, batchsize=batchsize, accumulation_steps=[(0, 1)], lr=0.0001, start_epoch=0, end_epoch=6, num_workers=12)
train.train(updata_utt=True)
