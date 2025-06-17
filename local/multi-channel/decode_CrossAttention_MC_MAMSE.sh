#!/usr/bin/env bash

stage=0
nj=8



data=
affix=
diarized_rttm=

model_path=

model_config=configs_mc_ca_4Speakers
max_speaker=4

oracle_vad=
train_set_speaker_embedding_list=exp/nnet3_recipe_ivector/ivectors_chime7_train_array_Oracle/ivectors_spk.txt
max_utt_durance=800
hop_len=600
batch_size=40
ivector_dir=exp/nnet3_recipe_ivector
do_vad=false
gpu=0,1,2,3

th_s=25
th_e=60

. ./utils/parse_options.sh
. path.sh
. cmd.sh

set -e

if [ $stage -le 0 ]; then
  mkdir -p data/$data
  find /disk1/chime/data/CHiME7/chime6/audio/dev | grep U > data/$data/wav.list
  awk -F '/' '{print $NF}' data/$data/wav.list | sed 's/\.wav//g' | sed 's/\./_/g' > data/$data/utt
  paste -d " " data/$data/utt data/$data/wav.list > data/$data/wav.scp
  paste -d " " data/$data/utt data/$data/utt > data/$data/utt2spk
  for l in `cat data/$data/utt`;do 
    s=`echo $l | cut -d "_" -f 1`
    grep $s $diarized_rttm | awk -v var=$l '{$2=var;print $0}'
  done > data/$data/diarized.rttm 
  cat /disk1/chime/data/CHiME7/chime6/oracle_rttm/dev/S* > data/$data/oracle.rttm
  utils/fix_data_dir.sh data/$data
fi

if [ $stage -le 1 ]; then
  rm -f data/$data/diarized.all.rttm
  for l in `awk '{print $1}' data/$data/wav.scp`;do
    s=$(echo $l | sed "s/\_CH.*//g")
    s=$(echo $s | sed "s/\_U.*//g")
    grep $s $diarized_rttm | awk -v l=$l '{$2=l;print $0}' \
      >> data/$data/diarized.all.rttm
  done
  local/extract_feature.sh --stage 3 --nj $nj \
      --sample_rate _16k --ivector_dir $ivector_dir \
      --max_speaker 4 --affix _$affix \
      --rttm data/$data/diarized.all.rttm --data $data
fi

if [ $stage -le 2 ]; then

    CUDA_VISIBLE_DEVICES=$gpu ../anaconda3/bin/python local/decode_CrossAttention_MC_MAMSE.py \
        --feature_list data/${data}/cmn_slide_fbank_htk.list \
        --embedding_list ${ivector_dir}/ivectors_${data}_${affix}/ivectors_spk.txt  \
        --train_set_speaker_embedding_list ${train_set_speaker_embedding_list} \
        --model_path ${model_path} \
        --output_dir ${model_path}_${data}_${affix} \
        --max_speaker $max_speaker \
        --init_rttm ${diarized_rttm} \
        --model_config ${model_config} \
        --max_utt_durance ${max_utt_durance} \
        --batch_size $batch_size \
        --set $data \
        --gpu $gpu \
        --hop_len $hop_len \
        --remove_overlap
fi

if  [ $stage -le 3 ]; then
  for th in `seq $th_s 5 $th_e`; do
    echo "$th "
    #if false;then
    ../anaconda3/bin/python local/postprocessing.py --threshold $th --medianfilter -1 \
        --prob_array_dir ${model_path}_${data}_$affix  --min_segments 0 \
        --min_dur 0.20 --segment_padding 0.30 --max_dur 0.10
        #--min_dur 0.20 --segment_padding 0.10 --max_dur 0.80
    local/analysis_diarization.sh data/$data/oracle.rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp 2>/dev/null | grep ALL
    
    if $do_vad;then
      python local/rttm_filter_with_vad.py \
          --input_rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp \
          --output_rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp_oraclevad \
          --oracle_vad ${oracle_vad}
      local/analysis_diarization.sh data/$data/fa.rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp_oraclevad 2>/dev/null | grep ALL
    fi
  done
fi
