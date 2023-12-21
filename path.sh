#export KALDI_ROOT=/yrfs1/intern/glzhong/kaldi
export KALDI_ROOT=/home/yoos/Documents/code/kaldi
export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export LD_LIBRARY_PATH=$KALDI_ROOT/src/lib:$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
#LD_LIBRARY_PATH=/yrfs5/sre/leisun8/tools/kaldi_cuda9/tools/sox/lib:$LD_LIBRARY_PATH
#PATH=/yrfs5/sre/leisun8/tools/kaldi_cuda9/tools/sox/bin:$PATH

#PATH=/home4/intern/rywang9/tools/sox/:$PATH
#LD_LIBRARY_PATH=/home4/intern/rywang9/tools/sox/lib:$LD_LIBRARY_PATH
#export PATH=/home/intern/stniu/anaconda3/bin/:$PATH
#export PATH=/home4/intern/stniu/anaconda3/envs/mss/bin:$PATH
#export PATH=/opt/lib/cuda-9.0_cudnn-v7.1.4/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/opt/lib/cuda-9.0_cudnn-v7.1.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export PATH=/opt/lib/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/lib/cudnn/cudnn-10.2-v7.6.5.32/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/work1/sre/leisun8/tools/libsndfile/lib/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home4/intern/mkhe/anaconda3/envs/torch/lib:$LD_LIBRARY_PATH
#. path_v100.sh
export PATH=/home4/intern/mkhe/anaconda3/bin/:$PATH
export LD_LIBRARY_PATH=/home4/intern/mkhe/anaconda3/lib:$LD_LIBRARY_PATH

export PATH=/home4/intern/stniu/libs/ffmpeg/bin/:$PATH
export LD_LIBRARY_PATH=/home4/intern/stniu/libs/ffmpeg/lib:$LD_LIBRARY_PATH
#CUDA_LAUNCH_BLOCKING=1
#export NCCL_IB_DISABLE=1
# NCCL_DEBUG=INFO

NCCL_SOCKET_IFNAME=eth0

#export PATH=/home3/cv1/hangchen2/anaconda3/envs/py38+cu102/bin/:$PATH
#export LD_LIBRARY_PATH=/home3/cv1/hangchen2/anaconda3/envs/py38+cu102/lib:$LD_LIBRARY_PATH
