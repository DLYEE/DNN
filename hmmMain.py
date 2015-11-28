import IO
import re
import math
import hmmFunc
import numpy as np

def viterbi():
    observationProb, keyOrder = IO.readFile('test.prb')
    outputData = {}
    label = IO.readLabel('data/label/train.lab')
    transMatrix = hmmFunc.transProb(label)
    value = []
    trace = []
    def forward(key, preValue, preTrace):
        for i in range(0,48):
            tempValue = [(math.log(preValue[prei]) +
                         math.log(transMatrix[prei][i]) +
                         math.log(observationProb[key][i]), 
                         preTrace[prei].append(IO.int2str(i))) for prei in range(0,48)]
            maxSequence = reduce(lambda x1,x2: x2 if x2[0] > x1[0] else x1, tempValue)  
            value[i] = maxSequence[0]
            trace[i] = maxSequence[1]
        return value, trace

    for index in range(len(keyOrder)):
        if index == 0:
            for i in range(0,48):
                value.append(observationProb[keyOrder[0]][i])
                trace.append([IO.int2str(i)])
        else:
            value, trace = forward(keyOrder[index], value, trace)
    maxIndex = argmax(value)
    print 'max value = ', value[maxIndex]
    print 'trace = ', trace[maxIndex]
    #preValue = [1.3212,....]
    #preTrace = [sil, d, s, ...]

viterbi()                                





