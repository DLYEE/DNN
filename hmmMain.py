import IO
import cPickle
import re

class viterbiNode:
    def __init__(self, trace, prob)
    self.trace = trace #record the trace in viterbi (sil, d, s...)
    self.prob = prob #record the current probability

def viterbi():
    inputData = {}
    outputData = {}
    keyOrder = []
    viterbiLayer = []
    currentSentenceName = ''
    sentence = []
    inputData, keyOrder = readPickle('data/fbank/test.ark','3lyr_4096nrn_1188in_prob_fixed_1.prb.test')
    for i in range(0, 48):
        viterbiLayer.append(viterbiNode([],1))
    for key in keyOrder:
        s = re.split("_| |\n", key)
        if s[:2] == currentSentenceName:
            sentence.append(key)
        else:
            currentSentenceName = s[:2]
            sentence = [key]
                    






