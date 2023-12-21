#!/usr/bin/env bash

# single model decode
bash /train8/sppro/gbyang/code/NSD-MS2S/local/decode_S2S_model.sh --stage 3 --data chime7_eval_all_CH --diarized_rttm data/chime7_eval_all_CH/f1.rttm --affix f1

# models fusion decode
bash /train8/sppro/gbyang/code/NSD-MS2S/local/decode_S2S_models_fusion.sh --stage 3 --data chime7_eval_all_CH --diarized_rttm data/chime7_eval_all_CH/f1.rttm --affix f1