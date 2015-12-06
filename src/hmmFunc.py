import IO
import re
import sys
import numpy as np

def transProb(label):
    transProbMatrix = np.zeros((48,48))
    totalCount = np.zeros(48)
    for i in range(0, len(label)):
        #count through each sentence
        for j in range(0, len(label[i])-1):
            transProbMatrix[label[i][j], label[i][j+1]] += 1
            totalCount[label[i][j]] += 1
    for i in range(0, 48):
        #divide the total num of each state for calculating the probibility
        transProbMatrix[i] = transProbMatrix[i] / totalCount[i]
        for j in range(0,48):
            #avoid 0 probibility
            transProbMatrix[i][j] = max(sys.float_info.min, transProbMatrix[i][j])
    return transProbMatrix

    

