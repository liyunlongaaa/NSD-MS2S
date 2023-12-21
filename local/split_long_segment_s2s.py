# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

def split_segment(prob, sess, spk, start, end, max_dur=2000):
    dur = end - start
    if dur <= max_dur:
        print("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>".format(sess, start/100., dur/100., spk))
    else:
        tosplit = int(start+100 + np.argmin(prob[int(start+100):int(end-100)]))
        split_segment(prob, sess, spk, start, tosplit)
        split_segment(prob, sess, spk, tosplit, end)


prob_array_dir = sys.argv[1]
input_rttm = sys.argv[2]
prob_array = [os.path.join(prob_array_dir, l) for l in os.listdir(prob_array_dir)]
prob_label = {}
#print(prob_array_dir, input_rttm)
for p in prob_array:
    if p.find(".npy") == -1: continue
    session = os.path.basename(p).split('.')[0]
    if session.find("CH") != -1 and session.find("S") != -1:
        sess = session.split("_")[0]
    elif session.find("CH") != -1 and session.find("S") == -1:
        sess = "_".join(session.split("_")[:-1])
    else:
        sess = session
    prob_label[sess] = np.load(os.path.join(p)) #num_spk, len
IN = open(input_rttm)
for l in IN:
    #print(l)
    line = l.split(" ")
    session = line[1]
    if line[-2] != "<NA>":
        spk = line[-2]
    else:
        spk = line[-3]
    #print(line[3] )
    start = np.int64(np.float64(line[3]) * 100 )
    dur =   np.int64(np.float64(line[4]) * 100)
    end = start + dur
    if dur <= 2000:
        print(l.rstrip())
        #pass
    else:
        split_segment(prob_label[session][int(spk)], session, spk, start, end, max_dur=2000)
