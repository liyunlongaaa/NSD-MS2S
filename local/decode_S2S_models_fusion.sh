#!/usr/bin/env bash
cd /train8/sppro/gbyang/code/NSD-MS2S
stage=3
nj=25

train_set_speaker_embedding_list=exp/nnet3_recipe_ivector/ivectors_chime7_train_array_Oracle/ivectors_spk.txt

model_type=model_S2S_weight_input_DIM

model_path=exp/S2S/Batchsize20_4speakers_Segment800s_Mixup0.5_CHiME6MAMSELabel_SimuCHiME6_Mixer6MAMSELabel_SimuMixer6_SimuDipcoDevNoise_all_data_512_all0Dropout_6layers_weight_input_DIM/MULTI_MAM_SE_S2S_model.model

epochs=0-6 # (0, 6] epoch model parameters fusion average


data=dipco_dev_all_CH
affix=f1
diarized_rttm=data/dipco_dev_all_CH/f1.rttm

model_config=N



oracle_vad=

max_utt_durance=800
max_speaker=4
batch_size=64
ivector_dir=exp/nnet3_recipe_ivector
do_vad=false
gpu=0,1

#post-processing parameter
hop_len=100
min_dur=0.2
segment_padding=0.10
max_dur=0.80
collar=true   # whether to use collar to calculate rttm, the default collar is 0.25

th_s=25
th_e=65

. path.sh
. utils/parse_options.sh

echo "${data} ${affix}"
echo "${model_path}"
# nvidia-smi
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
    CUDA_VISIBLE_DEVICES=$gpu python local/decode_MULTI_SE_MA_MSE_S2S_CH_fusion_models_fusion.py \
        --feature_list data/${data}/cmn_slide_fbank_htk.list \
        --embedding_list ${ivector_dir}/ivectors_${data}_${affix}/ivectors_spk.txt  \
        --train_set_speaker_embedding_list ${train_set_speaker_embedding_list} \
        --model_path ${model_path} \
        --output_dir ${model_path}_${data}_${affix}_fusion_models${epochs} \
        --max_speaker $max_speaker \
        --init_rttm ${diarized_rttm} \
        --model_config ${model_config} \
        --max_utt_durance ${max_utt_durance} \
        --hop_len ${hop_len} \
        --batch_size $batch_size \
        --gpu $gpu \
        --remove_overlap \
        --model_type $model_type \
        --epochs $epochs
fi

if  [ $stage -le 3 ]; then
  for th in `seq $th_s 5 $th_e`; do
    echo "$th "
    #if false;then
    python local/postprocessing_s2s.py --threshold $th --meanfilter -1 \
        --prob_array_dir ${model_path}_${data}_${affix}_fusion_models${epochs}  --min_segments 0 \
        --min_dur $min_dur --segment_padding $segment_padding --max_dur $max_dur
        # --min_dur 0.20 --segment_padding 0.20 --max_dur 0.80
        # --min_dur 0.20 --segment_padding 0.10 --max_dur 0.80
    # u=1
    # c=7
    #local/analysis_diarization.sh data/$data/oracle.rttm ${model_path}_${data}_$affix/U0${u}_CH$c.rttm 2>/dev/null
    #grep U0${u}_CH$c ${model_path}_${data}_$affix/rttm_th0.${th}_pp | sed "s/_U0${u}_CH$c//g" > ${model_path}_${data}_$affix/U0${u}_CH${c}_pp.rttm
    python local/split_long_segment_s2s.py ${model_path}_${data}_${affix}_fusion_models${epochs} ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp > ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp.tmp
    mv ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp.tmp ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp
    if $collar; then
      echo 'collar 0.25'
      local/analysis_diarization.sh --collar 0.25 data/$data/oracle.rttm ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp 
    else
      local/analysis_diarization.sh mamse/data/$data/oracle.rttm ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp 
    fi
    #local/analysis_diarization.sh chime6_rttm/dev_rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp 2>/dev/null
    if $do_vad;then
      python local/rttm_filter_with_vad.py \
          --input_rttm ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp \
          --output_rttm ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp_oraclevad \
          --oracle_vad ${oracle_vad}
      grep U0${u}_CH$c ${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.${th}_pp_oraclevad | sed "s/_U0${u}_CH$c//g" > ${model_path}_${data}_$affix/U0${u}_CH${c}_pp_oraclevad.rttm
      #local/analysis_diarization.sh data/$data/oracle.rttm ${model_path}_${data}_$affix/U0${u}_CH${c}_pp_oraclevad.rttm 2>/dev/null
      #local/analysis_diarization.sh data/$data/oracle.rttm ${model_path}_${data}_$affix/rttm_th0.${th}_pp_oraclevad 2>/dev/null
    fi
  done
fi

echo "${model_path}_${data}_${affix}_fusion_models${epochs}/rttm_th0.50_pp"
