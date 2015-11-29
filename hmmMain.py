import IO
import re
import math
import hmmFunc
import numpy as np
import time

def viterbi():
    observationProb, keyOrder = IO.readFile('test.prb')
    outputData = {}
    label = IO.readLabel('data/label/train.lab')
    transMatrix = hmmFunc.transProb(label)
    value = []
    trace = []
    def forward(key, preValue, preTrace):
        reValue = []
        reTrace = []
        for i in range(0,48):
            tempValue = [preValue[prei] +
                         math.log(transMatrix[prei][i]) +
                         math.log(observationProb[key][i]) for prei in range(0,48)]
            maxIndex = np.argmax(tempValue)
            reValue.append(tempValue[maxIndex])
            tempTrace = preTrace[maxIndex][:]
            reTrace.append(tempTrace)
            reTrace[i].append(IO.int2str(i))
        return reValue, reTrace
    
    timeStart = time.time()
    for index in range(len(keyOrder)):
        if index % 1000 == 0:
            timeEnd = time.time()
            print "Time Cost = ", timeEnd - timeStart
            timeStart = time.time()
        if index == 0:
            for i in range(0,48):
                value.append(math.log(observationProb[keyOrder[index]][i]))
                trace.append([IO.int2str(i)])
        else:
            value, trace = forward(keyOrder[index], value, trace)
    maxIndex = np.argmax(value)
    print 'max value = ', value[maxIndex]
    #preValue = [1.3212,....]
    #preTrace = [sil, d, s, ...]

viterbi()                                





