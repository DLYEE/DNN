import numpy as np
import time
import dnnTrainFunc
import IO

print ('batchsize =', dnnTrainFunc.batchSize)

def makeBatch(inputData, keyOrder, label, mode):
    inputBatches = []
    labelBatches = []
    for i in range(0, len(keyOrder), dnnTrainFunc.batchSize):
        if i <= len(keyOrder) - dnnTrainFunc.batchSize:
            inputBatch = np.asarray([ inputData[keyOrder[i+j]] for j in range(dnnTrainFunc.batchSize) ])
            inputBatches.append(inputBatch)
            if mode == 'train':
                labelBatch = np.asarray([ label[keyOrder[i+j]] for j in range(dnnTrainFunc.batchSize) ])
                labelBatches.append(labelBatch)
        elif mode == 'test':
            inputBatch = np.asarray([ inputData[keyOrder[i+j]] for j in range(len(keyOrder) - i) ])
            inputBatches.append(inputBatch)
    return inputBatches, labelBatches
###

def training(epochNum ,inputBatches, labelBatches):
    for epoch in range(epochNum):
        tStart = time.time()
        print ("Running the", epoch + 1, "th epoch...")
        cst = []
        grad = []
        for i in range(len(inputBatches)):
            zz = dnnTrainFunc.train(1, inputBatches[i].astype(dtype='float32'), labelBatches[i].astype(dtype='float32'))
            cst.append(zz[0])
            grad.append(zz[1:])
        print ("Cost = ", (np.mean(cst)/dnnTrainFunc.batchSize))
        print ("Gradient = ",(np.mean(grad)/dnnTrainFunc.batchSize))
        # print grad
        tEnd = time.time()
        print ("It cost %f sec" % (tEnd - tStart))

def testing(inputBatches, keyOrder):
    outputData = {}
    tOs = {}
    possibilityVectors= []

    for i in range(len(inputBatches)):
        tO = dnnTrainFunc.test(0, inputBatches[i].astype(dtype='float32'))      #testoutput
        index = dnnTrainFunc.batchSize * i                                      #'i' is the number of keyBatches & 'index' is the number of keyOrder
        for j in range(dnnTrainFunc.batchSize):
            if index+j < len(keyOrder):
                outputData[keyOrder[index+j]] = IO.int2str(np.argmax(tO[j]))
                tOs[keyOrder[index+j]] = tO[j]
    for index in range(len(keyOrder)):
        s = []
        s.append(keyOrder[index])
        s += tOs[keyOrder[index]]
        possibilityVectors.append(s)

    return outputData, possibilityVectors
