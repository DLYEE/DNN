import theano
import theano.tensor as T
import numpy as np
import random
import time
import IO

# InterNetwork: add 

class InterNetwork:

    def __init__(self, input, iNeuronNum, oNeuronNum, activateType, memoryValue):
        self._input = input
        self._weight = theano.shared((np.random.randn(oNeuronNum, iNeuronNum) / (iNeuronNum**0.5) ))
        self._memoryWeight = theano.shared((np.identity(oNeuronNum) ))
        self._bias = theano.shared(np.random.randn(oNeuronNum))
        self._activateType = activateType
        if self._activateType == 'SoftMax':
            self._output = T.dot(self._weight, self._input) + self._bias
        else:
            self._output = T.dot(self._weight, self._input) + self._bias + T.dot(self._memoryWeight, memoryValue)
        self._parameter = [self._weight, self._bias, self._memoryWeight]
        self._iNeuronNum = iNeuronNum
        self._oNeuronNum = oNeuronNum

    def activate(self) :
        if self._activateType == 'ReLU':
            # pass
            self._output = T.switch(self._output < 0, 0, self._output)

        elif self._activateType == 'SoftMax':
            MAX = T.max(self._output)
            self._output = self._output - MAX
            self._output = T.exp(self._output)
            self._output = self._output/T.sum(self._output)
        else:
            raise ValueError

class RNN:

    def __init__(self, input, a_tm1, layerSizes, lr):
    # layer_sizes is a np array of integers
        self._parameter         = []
        self._interNetworks      = []
        self._layerSizes        = layerSizes
        self._interNetworkNum    = len(layerSizes) - 1
        self._lr                = lr
        self._a_t = []

        self._interNetworks.append(
            InterNetwork(
                input = input,
                iNeuronNum = self._layerSizes[0],
                oNeuronNum = self._layerSizes[1],
                activateType = 'ReLU',
                memoryValue = a_tm1[0]
            )
        )
        self._interNetworks[0].activate()
        self._parameter.extend(self._interNetworks[0]._parameter)
        self._a_t.append(self._interNetworks[0]._output)

        for i in range(1, self._interNetworkNum-1):
            self._interNetworks.append(
                InterNetwork(
                    input = self._interNetworks[i-1]._output,
                    iNeuronNum = self._layerSizes[i],
                    oNeuronNum = self._layerSizes[i+1],
                    activateType = 'ReLU',
                    memoryValue = a_tm1[i]
                )
            )
            self._interNetworks[i].activate()
            self._parameter.extend(self._interNetworks[i]._parameter)
            self._a_t.append(self._interNetworks[i]._output)
        self._interNetworks.append(
                InterNetwork(
                    input = self._interNetworks[self._interNetworkNum-2]._output,
                    iNeuronNum = self._layerSizes[self._interNetworkNum-1],
                    oNeuronNum = self._layerSizes[self._interNetworkNum],
                    activateType = 'SoftMax',
                    memoryValue = a_tm1[self._interNetworkNum - 1]
                )
        )
        self._interNetworks[self._interNetworkNum-1].activate()
        self._parameter.extend(self._interNetworks[self._interNetworkNum-1]._parameter)
        self._a_t.append(self._interNetworks[self._interNetworkNum - 1]._output)
        self._output = self._interNetworks[self._interNetworkNum-1]._output

'''
    def update(self, gParameter):
    # Maintain root mean square of the gradient to update learning rate
    # update parameter set , movement set
        if len(self._rmgParameter) == 0 :
            self._rmgParameter = [T.max(gp, 1E-9) for gp in gParameter]
        else :
            self._rmgParameter = [ (rmgp ** 2 + gp ** 2) ** 0.5
                for rmgp , gp in zip(self._rmgParameter , gParameter) ]
        return [ (p,p - self._lr * gp / rmgp )
                for p, rmgp, gp in zip(self._parameter, self._rmgParameter, gParameter) ]
'''

layerSizes = [39, 128, 48]

a_0 = theano.shared(np.array([ np.zeros(layerSizes[i]) for i in range(1, len(layerSizes)-1) ]))
y_0 = theano.shared(np.zeros(layerSizes[len(layerSizes) - 1]))
print type(a_0), type(y_0)

x_seq = T.matrix()
y_hat_seq = T.matrix()
a_seq = T.tensor3()
y_seq = T.matrix()

def step(x_t, a_tm1, y_tm1):
    neuralNetwork = RNN(x_t, a_tm1, layerSizes, 1E-4)
    a_t = neuralNetwork._a_t
    y_t = neuralNetwork._output
    return a_t, y_t

[a_seq, y_seq],_ = theano.scan(
    step,
    sequences = x_seq,
    outputs_info = [a_0, y_0]
)

'''
cost = -T.log( neuralNetwork._output[0][T.argmax(labelFeature[0])] )
for i in xrange(1, batchSize):
    cost += -T.log( neuralNetwork._output[i][T.argmax(labelFeature[i])] )

grad = T.grad(cost, neuralNetwork._parameter)
'''
train = theano.function(
    on_unused_input = 'ignore',
    inputs = [x_seq, y_hat_seq],
    # outputs = [cost] + [g.norm(2) for g in grad],
    outputs = y_seq,
)

inputSequence ,keyOrder = RnnIO.readFile('possibility.txt', 'rnn')
labelSequence = RnnIO.readLabel('data/label/try1.lab', 48, 'rnn')

for epoch in range(1):
    tStart = time.time()
    print ("Running the", epoch + 1, "th epoch...")
    cst = []
    grad = []
    zz = train(inputSequence.astype(dtype='float32'), labelSequence.astype(dtype='float32'))
    print zz
    '''
    cst.append(zz[0])
    grad.append(zz[1:])
    print ("Cost = ", (np.mean(cst)/batchSize))
    print ("Gradient = ",(np.mean(grad)/batchSize))
    # print grad
    tEnd = time.time()
    print "It cost %f sec" % (tEnd - tStart)
    '''


