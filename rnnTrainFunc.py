
import theano
import theano.tensor as T
import numpy as np
import time
import rnnClass
import dnnClass
import IO

batchSize = 1
print ('batchsize =', batchSize)

def makeBatch(inputData, keyOrder, label, mode):
    global batchSize
    inputBatches = []
    labelBatches = []
    for i in range(0, len(keyOrder), batchSize):
        if i <= len(keyOrder) - batchSize:
            inputBatch = np.asarray([ inputData[keyOrder[i+j]] for j in range(batchSize) ])
            inputBatches.append(inputBatch)
            if mode == 'train':
                labelBatch = np.asarray([ label[keyOrder[i+j]] for j in range(batchSize) ])
                labelBatches.append(labelBatch)
        elif mode == 'test':
            inputBatch = np.asarray([ inputData[keyOrder[i+j]] for j in range(len(keyOrder) - i) ])
            inputBatches.append(inputBatch)
    return inputBatches, labelBatches
###


trainingMode = T.scalar()
x_seq = T.matrix()
y_hat_seq = T.matrix()
a_seq = T.tensor3()
y_seq = T.matrix()

layerSizes = [48, 128, 48]
def step(x_t, a_tm1List, y_tm1, cost_tm1, grad_tm1):
    neuralNetwork = RNN(trainingMode, x_t, layerSizes, 1E-4)
    a_tList = neuralNetwork.feedforward(a_tm1List)
    y_t = neuralNetwork._output[0]
    return a_tList, y_t

a_0 = theano.shared(np.array([ np.zeros(layerSizes[i]) for i in range(1, len(layerSizes)-1) ]))
y_0 = theano.shared(np.zeros(layerSizes[len(layerSizes) - 1]))
[a_seq, y_seq, cost],_ = theano.scan(
    step,
    sequences = x_seq,
    outputs_info = [a_0, y_0]
)

cost = -T.log( neuralNetwork._output[0][T.argmax(labelFeature[0])] )
grad = T.grad(cost_t, neuralNetwork._parameter)

train = theano.function(
    on_unused_input = 'ignore',
    inputs = [trainingMode, x_seq, y_hat_seq],
    outputs = [cost] + [g.norm(2) for g in grad],
    updates = neueralNetwork.update(grad)
)


train = theano.function(
    on_unused_input = 'ignore',
    inputs = [trainingMode, inputDataFeature, labelFeature],
    updates = neuralNetwork.update(grad),
    outputs = [cost] + [g.norm(2) for g in grad]
)

test = theano.function(
    on_unused_input = 'ignore',
    inputs = [trainingMode, inputDataFeature],
    outputs = neuralNetwork._output
)

def training(epochNum ,inputBatches, labelBatches):
    global batchSize
    for epoch in range(epochNum):
        tStart = time.time()
        print ("Running the", epoch + 1, "th epoch...")
        cst = []
        grad = []
        for i in range(len(inputBatches)):
            zz = train(1, inputBatches[i].astype(dtype='float32'), labelBatches[i].astype(dtype='float32'))
            cst.append(zz[0])
            grad.append(zz[1:])
        print ("Cost = ", (np.mean(cst)/batchSize))
        print ("Gradient = ",(np.mean(grad)/batchSize))
        # print grad
        tEnd = time.time()
        print ("It cost %f sec" % (tEnd - tStart))

def testing(inputBatches, keyOrder):
    global batchSize
    outputData = {}
    tOs = {}
    possibilityVectors= []

    for i in range(len(inputBatches)):
        tO = test(0, inputBatches[i].astype(dtype='float32'))               #testoutput
        index = batchSize * i                       #'i' is the number of keyBatches & 'index' is the number of keyOrder
        for j in range(batchSize):
            if index+j < len(keyOrder):
                outputData[keyOrder[index+j]] = IO.int2str(np.argmax(tO[j]))
                tOs[keyOrder[index+j]] = tO[j]
    for index in range(len(keyOrder)):
        s = []
        s.append(keyOrder[index])
        s += tOs[keyOrder[index]]
        possibilityVectors.append(s)

    return outputData, possibilityVectors

