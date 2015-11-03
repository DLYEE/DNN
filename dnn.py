import theano
import theano.tensor as T
import numpy as np
import random
import time
import IO

# Need to initialize values in line 105, 111, 134, 135, 162

class InterNetwork:

    def __init__(self, input, iNeuronNum, oNeuronNum, activateType, train):
        self._input = input
        self._weight = theano.shared((np.random.randn(oNeuronNum, iNeuronNum) / (iNeuronNum**0.5) ))
        self._bias = theano.shared(np.random.randn(oNeuronNum))
        self._activateType = activateType
        self._output = T.transpose(T.dot(self._weight, T.transpose(self._input))+self._bias.dimshuffle(0,'x'))
        self._parameter = [self._weight, self._bias]
        self._iNeuronNum = iNeuronNum
        self._oNeuronNum = oNeuronNum
        self._train = train

    def activate(self) :
        if self._activateType == 'ReLU':
            # pass
            self._output = T.switch(self._output < 0, 0, self._output)

            # dropout
            srng = T.shared_randomstreams.RandomStreams(seed = 123)
            if self._train == 1:
                self._output = self._output * srng.binomial(size = (self._oNeuronNum), p = 0.5)
            else:
                self._output = self._output * 0.5

        elif self._activateType == 'SoftMax':
            MAX = T.max(self._output,1)
            self._output = self._output - MAX.dimshuffle(0,'x')
            self._output = T.exp(self._output)
            self._output = self._output/T.sum(self._output,1).dimshuffle(0,'x')
        else:
            raise ValueError

class DNN:

    def __init__(self, training, input, layerSizes, lr):
    # layer_sizes is a np array of integers
        self._parameter         = []
        self._interNetworks      = []
        self._layerSizes        = layerSizes
        self._interNetworkNum    = len(layerSizes) - 1
        self._lr                = lr

        self._interNetworks.append(
            InterNetwork(
                input = input,
                iNeuronNum = self._layerSizes[0],
                oNeuronNum = self._layerSizes[1],
                activateType = 'ReLU',
                train = training
            )
        )
        self._interNetworks[0].activate()
        self._parameter.extend(self._interNetworks[0]._parameter)

        for i in range(1, self._interNetworkNum-1):
            self._interNetworks.append(
                InterNetwork(
                    input = self._interNetworks[i-1]._output,
                    iNeuronNum = self._layerSizes[i],
                    oNeuronNum = self._layerSizes[i+1],
                    activateType = 'ReLU',
                    train = training
                )
            )
            self._interNetworks[i].activate()
            self._parameter.extend(self._interNetworks[i]._parameter)
        self._interNetworks.append(
                InterNetwork(
                    input = self._interNetworks[self._interNetworkNum-2]._output,
                    iNeuronNum = self._layerSizes[self._interNetworkNum-1],
                    oNeuronNum = self._layerSizes[self._interNetworkNum],
                    activateType = 'SoftMax',
                    train = training
                )
        )
        self._interNetworks[self._interNetworkNum-1].activate()
        self._parameter.extend(self._interNetworks[self._interNetworkNum-1]._parameter)
        self._output = self._interNetworks[self._interNetworkNum-1]._output
        self._rmgParameter = []


    def update(self, gParameter):
    # Maintain root mean square of the gradient to update learning rate
    # update parameter set , movement set
        if len(self._rmgParameter) == 0 :
            self._rmgParameter = [T.max(gp, 1E-9) for gp in gParameter]
        else :
            self._rmgParameter = [ (rmgp ** 2 + gp ** 2) ** 0.5
                for rmgp , gp in zip(self._rmgParameter , gParameter) ]
        return [ (p,p - self._lr * gp / rmgp ) # Why / rmgp is wrong ?????
                for p, rmgp, gp in zip(self._parameter, self._rmgParameter, gParameter) ]

###
batchSize = 1

trainingMode = T.scalar('''dtype='float32' ''')
inputDataFeature = T.matrix('''dtype='float32' ''')
labelFeature = T.matrix('''dtype='float32' ''')
outputDataFeature = T.vector('''dtype='float32' ''')
neuralNetwork = DNN(trainingMode, inputDataFeature, [39, 128, 48], 1E-4)

cost = -T.log( neuralNetwork._output[0][T.argmax(labelFeature[0])] )
for i in xrange(1, batchSize):
    cost += -T.log( neuralNetwork._output[i][T.argmax(labelFeature[i])] )

grad = T.grad(cost, neuralNetwork._parameter)

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

###

inputData, keyOrder = IO.readFile('mfcc/try1.ark')
label = IO.readLabel('label/try1.lab', 48)

inputBatches = []
labelBatches = []
for i in xrange(0, len(keyOrder), batchSize):
    if i <= len(keyOrder) - batchSize:
        inputBatch = np.asarray([ inputData[keyOrder[i+j]] for j in xrange(batchSize) ])
        inputBatches.append(inputBatch)
        labelBatch = np.asarray([ label[keyOrder[i+j]] for j in xrange(batchSize) ])
        labelBatches.append(labelBatch)

###

for epoch in range(10):
    tStart = time.time()
    print ("Running the", epoch + 1, "th epoch...")
    # print T.sum(neuralNetwork._rmgParameter[0]).eval()
    # print neuralNetwork._parameter[0].get_value()
    cst = []
    grad = []
    for i in range(len(inputBatches)):
        zz = train(1, inputBatches[i], labelBatches[i])
        cst.append(zz[0])
        grad.append(zz[1:])
    print ("Cost = ", (np.mean(cst)/batchSize))
    print ("Gradient = ",(np.mean(grad)/batchSize))
    # print grad
    tEnd = time.time()
    print "It cost %f sec" % (tEnd - tStart)

inputData, keyOrder = IO.readFile('mfcc/try1.ark')
outputData = {}                             #outputDicr : dictionary of the output answer of test after int2str transfer
tOs = {}
possibilityVectors= []

inputBatches = []
for i in xrange(0, len(keyOrder), batchSize):
    if i <= len(keyOrder) - batchSize:
        inputBatch = np.asarray([ inputData[keyOrder[i+j]] for j in xrange(batchSize) ])
        inputBatches.append(inputBatch)
    else:
        inputBatch = np.asarray([ inputData[keyOrder[i+j]] for j in xrange(len(keyOrder) - i) ])
        inputBatches.append(inputBatch)

for i in range(len(inputBatches)):
    tO = test(0, inputBatches[i])               #testoutput
    index = batchSize * i                       #'i' is the number of keyBatches & 'index' is the number of keyOrder
    for j in xrange(batchSize):
        if index+j < len(keyOrder):
            outputData[keyOrder[index+j]] = IO.int2str(np.argmax(tO[j]))
            tOs[keyOrder[index+j]] = tO[j]
for index in range(len(keyOrder)):
    possibilityVectors.append(tOs[keyOrder[index]])

IO.writeFile('predict.csv', 'possibility.txt', possibilityVectors, outputData, keyOrder)
