import theano
import theano.tensor as T
import numpy as np
import time
import rnnClass
import dnnClass
import IO

batchSize = 5
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

def activate(output, activateType) :
        if activateType == 'ReLU':
            # pass
            output = T.switch(output < 0, 0, output)
            # dropout

        elif activateType == 'SoftMax':
            MAX = T.max(output,1)
            output = output - MAX.dimshuffle(0,'x')
            output = T.exp(output)
            output = output/T.sum(output,1).dimshuffle(0,'x')
        else:
            raise ValueError
###


trainingMode = T.scalar()
x_seq = T.matrix()
y_hat_seq = T.matrix()
a_seq = T.tensor3()
y_seq = T.matrix()
a_0 = theano.shared(np.zeros(layerSizes[1]))
y_0 = theano.shared(np.zeros(layerSizes[2]))
layerSizes = [48, 128, 48]
layerRange = T.vector()
neuralNetwork = RNN(trainingMode, x_seq, layerSizes, 1E-4)

def step(z_t, a_tm1):
    global neuralNetwork
    return activate(z_t + T.transepose(T.dot(neuralNetwork._memories[0]._weight, T.transpose(a_tm1))))

# feedforward
# first layer
z1_seq = T.transpose(T.dot(x_seq, neuralNetwork._intranets[0]._weight) + neuralNetwork._intranets[0]._bias.dimshuffle(0,'x'))
a_seq,_ = theano.scan(
        step,
        sequences = z1_seq,
        outputs_info = a_0,
        non_sequences = layerRange[0],
        truncate_gradient=-1
        )
# second layer
z2_seq = T.transpose(T.dot(a_seq, neuralNetwork._intranets[1]._weight) + neuralNetwork._intranets[1]._bias.dimshuffle(0,'x'))
y_seq,_ = theano.scan(
        step,
        sequences = z2_seq,
        outputs_info = a_0,
        non_sequences = layerRange[1],
        truncate_gradient=-1
        )
# output & cost/grad caculation
neuralNetwork._output = y_seq
cost = -T.log( y_seq[-1][T.argmax(y_hat_seq)] )
grad = T.grad(cost_t, neuralNetwork._parameter)

train = theano.function(
    on_unused_input = 'ignore',
    inputs = [trainingMode, x_seq, y_hat_seq, layerRange],
    outputs = [cost] + [g.norm(2) for g in grad],
    updates = neueralNetwork.update(grad)
)

test = theano.function(
    on_unused_input = 'ignore',
    inputs = [trainingMode, x_seq, layerRange],
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
            zz = train(1, inputBatches[i].astype(dtype='float32'), labelBatches[i].astype(dtype='float32'), range(1,3))
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
        tO = test(0, inputBatches[i].astype(dtype='float32'), range(1,3))               #testoutput
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
