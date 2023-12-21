#!/usr/bin/env bash
#

stage=1
nj=32
sample_rate=_16k
ivector_dir=exp/nnet3_recipe_ivector
data=
rttm=
max_speaker=4
affix=_Oracle
output_dir=./
# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./cmd.sh
. ./path.sh

#########################################################################
# Extract test data fbank
#########################################################################
if [ $stage -le 1 ]; then
  steps/make_fbank.sh --fbank-config conf/fbank${sample_rate}.conf --nj $nj \
    --cmd run.pl data/${data} $output_dir/exp/make_fbank/${data} $output_dir/fbank/${data}
fi

# # # #########################################################################
# # # # apply-cmvn-slide
# # # #########################################################################
write_num_frames_opt="--write-num-frames=ark,t:$output_dir/exp/make_fbank/$data/utt2num_frames.JOB"
real_data=`basename ${data}`
if [ $stage -le 2 ]; then
  #if [ ! -f data/${data}/cmn_slide_fbank_htk.list ]; then
    run.pl JOB=1:$nj $output_dir/exp/make_fbank/${data}/apply_cmn_slide_feats.JOB.log \
      apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 \
      scp:$output_dir/fbank/${data}/raw_fbank_${real_data}.JOB.scp ark:- \| \
      copy-feats $write_num_frames_opt ark:- \
      ark,scp:$output_dir/fbank/${data}/cmn_slide_fbank.JOB.ark,$output_dir/fbank/${data}/cmn_slide_fbank.JOB.scp || exit 1;
    
    # Convert ark,scp to htk
    rm -rf $output_dir/fbank/HTK/${data}_cmn_slide
    mkdir -p $output_dir/fbank/HTK/${data}_cmn_slide
    run.pl JOB=1:$nj $output_dir/exp/make_fbank/${data}/copy-feats-to-htk.JOB.log \
        copy-feats-to-htk --output-dir=$output_dir/fbank/HTK/${data}_cmn_slide \
        scp:$output_dir/fbank/${data}/cmn_slide_fbank.JOB.scp
    find $output_dir/fbank/HTK/${data}_cmn_slide | grep fea > \
      data/${data}/cmn_slide_fbank_htk.list
  #fi  
fi

# # # # #########################################################################
# # # # # Extract test data ivector with diarized rttm for evaluation 
# # # # #########################################################################
if [ $stage -le 3 ]; then
  if [ -f $rttm ]; then
    local/extract_ivector_session_level.sh --nj $nj --stage 0 \
      --silence_weight 0 --overlap_weight 0  --max-count 75 \
      --sample_rate ${sample_rate} --rttm $rttm --ivector-affix $affix \
      --output_dir $output_dir --max_speaker $max_speaker \
      data/${data} ${ivector_dir}
  else
    echo "$rttm not exist!"
  fi
fi

