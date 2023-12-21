# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse
import copy


def get_oracle_vad_mask(path_vad):
    session2vad = {}
    session2vad_list = {}
    T = 180 * 60 * 100
    for line in open (path_vad):
        session = line.split(' ')[0]
        start = line.split(' ')[1]
        end = line.split(' ')[2]
        if (session not in session2vad):
            session2vad[session] = np.zeros(T)
            session2vad_list[session] = []
        session2vad_list[session].append([int(float(start)*100), int(float(end)*100)])
        session2vad[session][int(float(start)*100): int(float(end)*100)] = 1
    return session2vad, session2vad_list

def get_rttm(input_path):
    rttm = {}
    T = 180 * 60 * 100
    with open(input_path) as INPUT:
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
                rttm[session][spk] = np.zeros(T)
            #print(line[3] )
            start = np.int64(np.float64(line[3]) * 100 )
            end = start + np.int64(np.float64(line[4]) * 100)
            rttm[session][spk][start:end] = 1
    return rttm

def write_rttm(session_label, output_path):
    with open(output_path, "w") as OUT:
        for session in session_label.keys():
            for spk in session_label[session].keys():
                labels = session_label[session][spk]
                to_split = np.nonzero(labels[1:] != labels[:-1])[0]
                to_split += 1
                if labels[-1] == 1:
                    to_split = np.r_[to_split, len(labels)+1]
                if labels[0] == 1:
                    to_split = np.r_[0, to_split]
                for l in to_split.reshape(-1, 2):
                    #print(l)
                    #break
                    OUT.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(session, l[0]/100., (l[1]-l[0])/100., spk))

def process(input_rttm, oracle_vad, dur = 30):
    rttm = get_rttm(input_rttm)
    oracle_vad, _ = get_oracle_vad_mask(oracle_vad)
    for session in rttm.keys():
        silence = 0
        real_session = session.split("_")[0]
        for spk in rttm[session].keys():
            silence += rttm[session][spk]
        try:
            for spk in rttm[session].keys():
                rttm[session][spk] = rttm[session][spk] * oracle_vad[session]
            oracle_vad[session][silence>0] = 0
            labels = oracle_vad[session]
        except:
            for spk in rttm[session].keys():
                rttm[session][spk] = rttm[session][spk] * oracle_vad[real_session]
            labels = copy.deepcopy(oracle_vad[real_session])
            labels[silence>0] = 0
        to_split = np.nonzero(labels[1:] != labels[:-1])[0]
        to_split += 1
        if labels[-1] == 1:
            to_split = np.r_[to_split, len(labels)+1]
        if labels[0] == 1:
            to_split = np.r_[0, to_split]
        for l in to_split.reshape(-1, 2):
            max_len = -1
            for spk in rttm[session].keys():
                cur_len = np.sum(rttm[session][spk][(l[0]-dur):l[0]]) + np.sum(rttm[session][spk][l[1]:(l[1]+dur)])
                if cur_len > max_len:
                    max_spk = spk
                    max_len = cur_len
            rttm[session][max_spk][l[0]:l[1]] = 1
    return rttm
    


def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')
    parser.add_argument('--input_rttm', metavar='PATH', type=str, required=True,
                        help='rttm.')
    parser.add_argument('--output_rttm', metavar='PATH', type=str, required=True,
                        help='rttm.')
    parser.add_argument('--oracle_vad', metavar='PATH', type=str, required=True,
                        help='oracle_vad.')
    parser.add_argument('--dur', metavar='PATH', type=int, default=15,
                        help='dur.')
    return parser

def main(args):
    '''
    rttm = get_rttm(args.input_rttm)
    oracle_vad, oracle_vad_list = get_oracle_vad_mask(args.oracle_vad)
    for session in rttm.keys():
        silence = 0
        for spk in rttm[session].keys():
            silence += rttm[session][spk]
        for spk in rttm[session].keys():
            rttm[session][spk] = rttm[session][spk] * oracle_vad[session]
            #pass
        dur = 30
        for l in oracle_vad_list[session]:
            for start in range(l[0], l[1], dur):
                ends = min(start+dur, l[1])
                if np.sum(silence[start:ends]==0) > 0:
                    max_spk = spk
                    max_len = np.sum(rttm[session][spk][start:ends])
                    for spk in rttm[session].keys():
                        if np.sum(rttm[session][spk][start:ends]) > max_len:
                            max_spk = spk
                            max_len = np.sum(rttm[session][spk][start:ends])
                    rttm[session][max_spk][start:ends][silence[start:ends]==0] = 1
    '''
    rttm = process(args.input_rttm, args.oracle_vad, args.dur)
    write_rttm(rttm, args.output_rttm)

if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
