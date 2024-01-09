#!/usr/bin/env bash
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

silence_weight=0
overlap_weight=0
stage=0
nj=8
ivector_affix=
sub_speaker_frames=0
max_count=75
rttm=
prob_dir=
conf_dir=conf
min_segment_length=10
max_speaker=8
sample_rate=
output_dir=./
# Begin configuration section.
# End configuration section
. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <data-dir> <ivector-dir>"
  echo -e >&2 "eg:\n  $0 data/train_data exp/nnet3_cleaned/extrctor"
  exit 1
fi

data_dir=$1
ivector_dir=$2

set -e -o pipefail

data_set=`basename ${data_dir}`
ivector_process_dir=${data_dir}/ivector${ivector_affix}
#######################################################################
# Prepare ivector dir for libricss data
#######################################################################
if [ $stage -le 0 ]; then
  rm -rf ${ivector_process_dir}
  if [ -f "$rttm" ]; then
    python local/prepare_ivector_extractor_dir_with_rttm.py \
      --silence_weight ${silence_weight} \
      --overlap_weight ${overlap_weight} --datadir ${data_dir} \
      --ivector_dir ${ivector_process_dir} --rttm $rttm \
      --max_speaker ${max_speaker} --min_segment_length ${min_segment_length}
  else
    echo "Need rttm or prob_dir for ivector extraction"
    exit 1
  fi
  utils/fix_data_dir.sh ${ivector_process_dir}
  # extract_ivector.sh needs lang_dir for creat weights, 
  # but in this part we use the existed weights, 
  # so we creat a fake lang dir for extract_ivector.sh checking
  mkdir -p ${ivector_process_dir}/lang/phones
  echo "None" > ${ivector_process_dir}/lang/phones.txt
  echo "None" > ${ivector_process_dir}/lang/phones/silence.csl
fi


#######################################################################
# compute mfcc for extracting ivectors
#######################################################################
if [ $stage -le 1 ]; then
  if [ ! -s ${ivector_process_dir}/feats.scp ]; then
    steps/make_mfcc.sh --mfcc-config ${conf_dir}/mfcc_hires${sample_rate}.conf --nj $nj \
      --cmd "$train_cmd" ${ivector_process_dir} $output_dir/exp/make_mfcc/$data_set $output_dir/mfcc/$data_set
    steps/compute_cmvn_stats.sh ${ivector_process_dir}
    utils/fix_data_dir.sh ${ivector_process_dir}

    # some sort segments will be ignored when compute mfcc
    # filter those segments' weights
    utils/filter_scp.pl ${ivector_process_dir}/utt2spk \
      ${ivector_process_dir}/weights > ${ivector_process_dir}/weights.filter
    sort ${ivector_process_dir}/weights.filter > ${ivector_process_dir}/weights
    gzip -c ${ivector_process_dir}/weights > ${ivector_process_dir}/weights.gz
  fi
fi


#######################################################################
# extract ivector
#######################################################################
if [ $stage -le 2 ]; then
  steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
    --ivector-period 10 --sub-speaker-frames $sub_speaker_frames \
    --max-count $max_count \
    ${ivector_process_dir} ${ivector_process_dir}/lang $ivector_dir/extractor \
    ${ivector_process_dir}/weights.gz \
    $ivector_dir/ivectors_${data_set}${ivector_affix}
fi


#######################################################################
# here save the ivector in an text file with following format:
# speaker1 [ ivector1 ]
# speaker2 [ ivector2 ]
# ...
#######################################################################
if [ $stage -le 3 ]; then
  echo "convert ivector format"
  for i in `seq 1 $nj`; do
    copy-feats ark:$ivector_dir/ivectors_${data_set}${ivector_affix}/ivectors_spk.$i.ark ark,t:-
  done > $ivector_dir/ivectors_${data_set}${ivector_affix}/ivectors_spk.txt
fi
