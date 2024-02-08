# -*- coding: utf-8 -*-

import numpy as np
import os
import argparse
import torch

def median_filtering(prob_label, padding="reflect", taps=51):
    filtered = {}
    signal = prob_label
    if signal.shape[1] <= taps//2:
        return signal 
    pad_signal = np.pad(signal, ((0, 0), (taps//2, taps//2)), padding)
    filterd_signal = np.zeros(signal.shape)
    # print(pad_signal.shape)
    for i in range(taps//2, taps//2+signal.shape[1]):
        # print(np.median(pad_signal[i -taps//2:i +taps//2],axis =1))
        filterd_signal[:, i-taps//2] = np.median(pad_signal[:, i-taps//2: i+taps//2+1], axis=1)
    filtered = np.array(filterd_signal)
    return filtered

def mean_filtering(prob_label, padding="reflect", taps=51):
    #filtered = {}
    avgpool = torch.nn.AvgPool1d(taps, 1)
    #for session in session_label.keys():
    signal = prob_label
    #print(signal.shape)
    pad_signal = np.pad(signal, ((0, 0), (taps//2, taps//2)), padding)
    filtered = avgpool(torch.from_numpy(pad_signal)[None, ...])[0, ...].numpy()
    #print(filtered[session].shape)
    return filtered

def write_rttm(session_label, output_path, min_segments, frame_per_second=100):
    with open(output_path, "w") as OUT:
        for session in session_label.keys():
            for spk in range(len(session_label[session])):
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
                    if (l[1]-l[0])/100. < min_segments:
                        continue
                    OUT.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(session, l[0]*1.0/frame_per_second, (l[1]-l[0])*1.0/frame_per_second, spk))

class Segmentation(object):
    """Stores segmentation for an utterances"""
    "this code comes from steps/segmentation/internal/sad_to_segments.py (reversed)"
    def __init__(self):
        self.segments = None
        self.filter_short_duration = 0

    def initialize_segments(self, rttm):
        self.segments = {}
        with open(rttm, 'r') as INPUT:
            for line in INPUT:
                line = line.split(" ")
                for l in line:
                    if l == "": line.remove(l)
                session = line[1]
                if line[-2] != "<NA>":
                    spk = line[-2]
                else:
                    spk = line[-3]
                if not session in self.segments.keys():
                    self.segments[session] = {}
                if not spk in self.segments[session].keys():
                    self.segments[session][spk] = []
                start = float(line[3])
                end = start + float(line[4])
                self.segments[session][spk].append([start, end])

    def filter_short_segments(self, min_dur):
        """Filters out segments with durations shorter than 'min_dur'."""
        if min_dur <= 0:
            return

        segments_kept = {}
        for session in self.segments.keys():
            segments_kept[session] = {}
            for spk in self.segments[session].keys():
                segments_kept[session][spk] = []
                for segment in self.segments[session][spk]:
                    dur = segment[1] - segment[0]
                    if dur < min_dur:
                        self.filter_short_duration += dur
                    else:
                        segments_kept[session][spk].append(segment)
        self.segments = segments_kept

    def pad_speech_segments(self, segment_padding, max_duration=float("inf")): 
        """Pads segments by duration 'segment_padding' on either sides, but
        ensures that the segments don't go beyond the neighboring segments
        or the duration of the utterance 'max_duration'."""
        if segment_padding <= 0:
            return
        if max_duration == None:
            max_duration = float("inf")
        
        for session in self.segments.keys():
            for spk in self.segments[session].keys():
                for i, segment in enumerate(self.segments[session][spk]):
                    segment[0] -= segment_padding  # try adding padding on the left side
                    if segment[0] < 0.0:
                        # Padding takes the segment start to before the beginning of the utterance.
                        # Reduce padding.
                        segment[0] = 0.0
                    if i >= 1 and self.segments[session][spk][i - 1][1] > segment[0]:  
                        # Padding takes the segment start to before the end the previous segment.
                        # Reduce padding.
                        segment[0] = self.segments[session][spk][i - 1][1]  

                    segment[1] += segment_padding
                    if segment[1] >= max_duration:
                        # Padding takes the segment end beyond the max duration of the utterance.
                        # Reduce padding.
                        segment[1] = max_duration
                    if (i + 1 < len(self.segments[session][spk])
                            and segment[1] > self.segments[session][spk][i + 1][0]):
                        # Padding takes the segment end beyond the start of the next segment.
                        # Reduce padding.
                        segment[1] = self.segments[session][spk][i + 1][0]

    def merge_consecutive_segments(self, max_dur): 
        """Merge consecutive segments (happens after padding), provided that
        the merged segment is no longer than 'max_dur'."""
        if max_dur <= 0 or not self.segments:
            return

        merged_segments = {}
        for session in self.segments.keys():
            merged_segments[session] = {}
            for spk in self.segments[session].keys():
                try:
                    merged_segments[session][spk] = [self.segments[session][spk][0]]
                except:
                    print(f"{session}-{spk}")
                    continue
                for segment in self.segments[session][spk][1:]:
                    #if segment[0] == merged_segments[session][spk][-1][1] and \
                    #        segment[1] - merged_segments[session][spk][-1][0] <= max_dur:
                    if segment[0] - merged_segments[session][spk][-1][1] <= max_dur:
                        # The segment starts at the same time the last segment ends,
                        # and the merged segment is shorter than 'max_dur'.
                        # Extend the previous segment.
                        merged_segments[session][spk][-1][1] = segment[1]
                    else:
                        merged_segments[session][spk].append(segment)

        self.segments = merged_segments

    def write_rttm(self, output_path):
        OUTPUT = open(output_path, 'w')
        for session in self.segments.keys():
            for spk in self.segments[session].keys():
                for segment in self.segments[session][spk]:
                    OUTPUT.write("SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n".format(session, segment[0], segment[1]-segment[0], spk))

def main(args):
    if os.path.isdir(args.prob_array_dir):
        prob_array = [os.path.join(args.prob_array_dir, l) for l in os.listdir(args.prob_array_dir)] 
    else:
        prob_array = [l.rstrip() for l in open(args.prob_array_dir)]
        args.prob_array_dir = os.path.dirname(args.prob_array_dir)
    session_label = {}
    for p in prob_array:
        if p.find(".npy") == -1: continue
        prob_label = np.load(os.path.join(p))
        session = os.path.basename(p).split('.')[0]

        num_speaker, T = prob_label.shape
        session_label[session] = np.zeros([num_speaker, T])
        if args.medianfilter != -1: 
            prob_label = median_filtering(prob_label, taps=args.medianfilter)
        if args.meanfilter != -1:  
            prob_label = mean_filtering(prob_label, taps=args.meanfilter)
        session_label[session][prob_label > args.threshold / 100] = 1  
        
    write_rttm(session_label, os.path.join(args.prob_array_dir, "rttm_th{:.2f}".format(args.threshold / 100)), args.min_segments, args.frame_per_second)
   
    segmentation = Segmentation()
    segmentation.initialize_segments(os.path.join(args.prob_array_dir, "rttm_th{:.2f}".format(args.threshold / 100)))
    segmentation.filter_short_segments(args.min_dur)
    segmentation.pad_speech_segments(args.segment_padding)
    segmentation.merge_consecutive_segments(args.max_dur)
    segmentation.write_rttm(os.path.join(args.prob_array_dir, "rttm_th{:.2f}_pp".format(args.threshold / 100)))


def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')
    parser.add_argument('--threshold', metavar='Float', type=float, default=0.5,
                        help='threshold.') 
    parser.add_argument('--min_dur', metavar='Float', type=float, default=0.0,
                        help='min_dur.') 
    parser.add_argument('--segment_padding', metavar='Float', type=float, default=0.0,
                        help='segment_padding.') 
    parser.add_argument('--max_dur', metavar='Float', type=float, default=0.0,
                        help='max_dur.') 
    parser.add_argument('--prob_array_dir', metavar='DIR', required=True,
                        help='prob_array_dir.')
    parser.add_argument('--meanfilter', metavar='Int', type=int, default=-1,
                        help='meanfilter.')       
    parser.add_argument('--medianfilter', metavar='Int', type=int, default=-1,
                        help='medianfilter.')        
    parser.add_argument('--min_segments', metavar='DIR', type=int, default=0,
                        help='min_segments.')
    parser.add_argument('--oracle_vad', metavar='PATH', type=str, default="",
                        help='min_segments.')
    parser.add_argument('--frame_per_second', metavar='int', type=float, default=100,
                        help='frame_per_second.') 
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)

