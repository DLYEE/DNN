import IO
import re
import math
import hmmFunc
import numpy as np
import time

def viterbi():
    observationProb, keyOrder, length = IO.readFile('data/test.prb')
    outputData = {}
    trainLabel = IO.readTrainLabel('data/label/train.lab')
    transMatrix = hmmFunc.transProb(trainLabel)

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

    names = []
    sentences = []
    count = -1
    # for sentence in range(1, 4):
    for sentence in range(1, len(length)):
        print "This sentence starts at ", keyOrder[count+1]
        print "Sentence Length = ", length[sentence] - length[sentence - 1]
        print "Toatal Length = ", length[sentence]
        timeStart = time.time()
        trace = []
        value = []
        for index in range(length[sentence] - length[sentence - 1]):
            count += 1
            if index == 0:
                for i in range(0,48):
                    value.append(math.log(observationProb[keyOrder[count]][i]))
                    trace.append([IO.int2str(i)])
            else:
                value, trace = forward(keyOrder[count], value, trace)
        timeEnd = time.time()
        print "Time Cost = ", timeEnd - timeStart
        maxIndex = np.argmax(value)
        print "max trace: ", trace[maxIndex]
        print 'max value = ', value[maxIndex]
        s = re.split('_', keyOrder[count])[:2]
        names.append(s[0] + '_' + s[1])
        sentences.append(trace[maxIndex])
        #preValue = [1.3212,....]
        #preTrace = [sil, d, s, ...]
    return names, sentences

names, sentences = viterbi()
IO.writeFile('123.csv', names, sentences)
IO.trimOutput('123.csv', 'hmm.csv')






