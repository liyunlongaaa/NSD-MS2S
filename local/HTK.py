# -*- coding: utf-8 -*-

import numpy
import struct


def readHtk(filename):
    '''
    Reads the features in a HTK file, and returns them in a 2-D numpy array.
    '''
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        # sampPeriod and parmKind will be omitted
        # Read data
        data = struct.unpack(">%df" % (nSamples * sampSize / 4), f.read(nSamples * sampSize))
        # return numpy.array(data).reshape(nSamples, int(sampSize / 4))
        return nSamples, sampPeriod, sampSize, parmKind, data

def readHtk_start_end(filename, start, end):
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        # sampPeriod and parmKind will be omitted
        f.seek(start * sampSize,1)
        # Read data
        data = struct.unpack(">%df" % ((end - start) * sampSize / 4), f.read((end - start) * sampSize))
        # return numpy.array(data).reshape(nSamples, int(sampSize 1 4))
        return nSamples, sampPeriod, sampSize, parmKind, data

def readHtk_info(filename):
    with open(filename, "rb") as f:
        # Read header
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
        return nSamples, sampPeriod, sampSize, parmKind

def writeHtk(filename, feature, sampPeriod=3200, parmKind=9):
    '''
    Writes the features in a 2-D numpy array into a HTK file.
    '''
    with open(filename, "wb") as f:
        # Write header
        nSamples = feature.shape[0]
        sampSize = feature.shape[1] * 4
        f.write(struct.pack(">iihh", nSamples, sampPeriod, sampSize, parmKind))
        # Write data
        f.write(struct.pack(">%df" % (nSamples * sampSize / 4), *feature.ravel()))
        
        
def read_wav_start_end(path, start, end):
    dur = end - start
    with open(path, "rb") as f:
        f.seek(44 + start * 2, 1)
        data = struct.unpack("<%dh" % (dur), f.read(dur*2))
    #print(dur, numpy.array(data).shape)
    return numpy.array(data) / 32768.