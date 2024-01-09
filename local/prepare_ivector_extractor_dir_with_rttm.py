# -*- coding: utf-8 -*-

'''
Prepare a data dir for ivector extract. Treat a speaker from librispeech speech as different speakers in each session.
'''

import os
import argparse
import numpy as np

np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress = True)

def main(args):
    if not os.path.isdir(args.ivector_dir):
        os.makedirs(args.ivector_dir)
    utt2spk = open(os.path.join(args.ivector_dir, "utt2spk"), 'a')
    weights = open(os.path.join(args.ivector_dir, "weights"), 'a')
    segments = open(os.path.join(args.ivector_dir, "segments"), 'a')
    rttm = {}
    MAX_len = {}
    with open(args.rttm, 'r') as INPUT:
        print('rttm', args.rttm)
        for line in INPUT:
            '''
            SPEAKER session0_CH0_0S 1 417.315   9.000 <NA> <NA> 1 <NA> <NA>
            '''
            line = line.split(" ")
            while "" in line:
                line.remove("")
            session = line[1]
            start = np.int64(np.float64(line[3]) * 100 )
            end = start + np.int64(np.float64(line[4]) * 100)
            if session not in MAX_len.keys():
                MAX_len[session] = 0
            if MAX_len[session] < end:
                MAX_len[session] = end
    with open(args.rttm, 'r') as INPUT:
        for line in INPUT:
            '''
            SPEAKER session0_CH0_0S 1 417.315   9.000 <NA> <NA> 1 <NA> <NA>
            '''
            line = line.split(" ")
            while "" in line:
                line.remove("")
            session = line[1]
            if not session in rttm.keys() :
                rttm[session] = {}
            if line[-2] != "<NA>":
                spk = line[-2]
            else:
                spk = line[-3]
            if not spk in rttm[session].keys():
                rttm[session][spk] = np.zeros(MAX_len[session])
            #print(line[3] )
            start = np.int64(np.float64(line[3]) * 100 )
            end = start + np.int64(np.float64(line[4]) * 100)
            rttm[session][spk][start:end] = 1
    nonoverlap = {}
    for session in rttm.keys():
        if args.remove_overlap:
            num_speaker = 0
            for spk in rttm[session].keys():
                num_speaker += rttm[session][spk]
            for spk in rttm[session].keys():
                rttm[session][spk][num_speaker!=1] = 0
        speaker_duration = []
        speakers = []
        for spk in sorted(list(rttm[session].keys())):
            speaker_duration.append(np.sum(rttm[session][spk])) # speaker order 1,2,3,4
            speakers.append(spk)
        #if len(speaker_duration) < args.max_speaker:
        #    print("number speaker of a session < {}.".format(args.max_speaker))
        #    exit(1)
        speaker_duration_id_order = sorted(list(range(len(speaker_duration))), reverse=True, key=lambda k:speaker_duration[k])
        for spk_idx in speaker_duration_id_order[:args.max_speaker]:
            spk = speakers[spk_idx]
            session_spk = "{0:}-{1:}".format(session, spk)
            i = 0
            while i < len(rttm[session][spk]):
                if rttm[session][spk][i] == 1:
                    start, durance = i, 1
                    i += 1
                    while i < len(rttm[session][spk]) and rttm[session][spk][i] == 1:
                        i += 1
                        durance += 1
                    if durance < args.min_segment_length:
                        continue
                    utt_id = "{}-{:06d}-{:06d}".format(session_spk, start, start+durance)
                    utt2spk.write("{} {}\n".format(utt_id, session_spk))
                    segments.write("{} {} {} {}\n".format(utt_id, session, start / 100., start / 100. + durance / 100.))
                    spk_weights = rttm[session][spk][start:start+durance] * 1.0
                    weights.write("{} {}\n".format(utt_id, list(spk_weights)).replace(",", "").replace("[", "[ ").replace("]", " ]"))
                i += 1
    utt2spk.close()
    weights.close()
    segments.close()
    os.system("cp {} {}".format(os.path.join(args.datadir, "wav.scp"), args.ivector_dir))
    


def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')
    parser.add_argument('--silence_weight', metavar='Float', type=float, required=True,
                        help='silence_weight.')
    parser.add_argument('--overlap_weight', metavar='Float', type=float, required=True,
                        help='Datadir directory name.')
    parser.add_argument('--datadir', metavar='DIR', required=True,
                        help='Datadir directory name.')
    parser.add_argument('--ivector_dir', metavar='DIR', required=True,
                        help='Datadir directory name.')
    parser.add_argument('--rttm', metavar='path', required=True,
                        help='Datadir directory name.')
    parser.add_argument('--remove_overlap', metavar='bool', type=bool, default=True,
                        help='Datadir directory name.')
    parser.add_argument('--max_speaker', metavar='bool', type=int, default=8,
                        help='Datadir directory name.')  
    parser.add_argument('--min_segment_length', metavar='INT', type=int, default=0,
                        help='Datadir directory name.')    # 0.1s = 10 / 100
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)