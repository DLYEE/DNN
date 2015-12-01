import IO
import re
import sys
import numpy as np

'''
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
                    
# def transProb(label):
    # startProb = []
    # endProb = []
    # succesiveProb = []
    # #y0 = aa, y1 = aa...followed the order in str2int
    # #startProb = [P(y0|start), P(y1|start)...]
    # #endProb = [P(y0|end), P(y1|end)...]
    # #succesiveProb = [[P(y0|y0), P(y1|y0)...],[P(y0|y1),P(y1,y1)...]...]
    # sequenceName = ''
    # currentLabel = ''
    # lastLabel = ''
    # startCount = np.zeros(49) #startCount[48] is for total number
    # endCount = np.zeros(49)
    # succesiveCount = np.zeros((48,49))
    # labelFile = open(label, 'r+')
    # while True:
        # line = labelFile.readline()
        # if not line:
            # endCount[48] += 1
            # endCount[str2int(lastLabel)] += 1
            # break        
        # s = re.split(',|_|\n',line)  
        # #s[0] = maeb0, s[1] = sil411, s[2] = 1, s[3] = sil
        # currentLabel = s[3]
        # if sequenceName != (s[0] + s[1]):
            # if sequenceName != '':
                # endCount[48] += 1
                # endCount[str2int(lastLabel)] += 1
            # sequenceName = s[0] + s[1]
            # startCount[48] += 1
            # startCount[str2int(currentLabel)] += 1
        # else:
            # succesiveCount[str2int(lastLabel)][str2int(currentLabel)] += 1
            # succesiveCount[str2int(lastLabel)][48] += 1
        # lastLabel = s[3]
    # for i in range(0,48):
        # subProb = []
        # startProb.append(float(startCount[i]/max(startCount[48],1)))
        # endProb.append(float(endCount[i]/max(endCount[48],1)))
        # for j in range(0,48):
            # subProb.append(float(succesiveCount[i][j]/max(succesiveCount[i][48],1)))
        # succesiveProb.append(subProb)
'''
def transProb(label):
    transProbMatrix = np.zeros((48,48))
    totalCount = np.zeros(48)
    for i in range(0, len(label)):
        for j in range(0, len(label[i])-1):
            transProbMatrix[label[i][j], label[i][j+1]] += 1
            totalCount[label[i][j]] += 1
    for i in range(0, 48):
        transProbMatrix[i] = transProbMatrix[i] / totalCount[i]
        for j in range(0,48):
            transProbMatrix[i][j] = max(sys.float_info.min, transProbMatrix[i][j])
    return transProbMatrix

    

