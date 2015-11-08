import theano
import theano.tensor as T
import numpy as np
import random

# Need to initialize values in line 105, 111, 134, 135, 162

class InterNetwork:

    def __init__(self, iNeuronNum, oNeuronNum, activateType):
        # self._input = []
        self._weight = theano.shared(np.random.randn(oNeuronNum, iNeuronNum).astype(dtype='float32') / (iNeuronNum**0.5) )
        self._bias = theano.shared(np.random.randn(oNeuronNum).astype(dtype='float32'))
        self._activateType = activateType
        self._output = []
        self._parameter = [self._weight, self._bias]
        self._iNeuronNum = iNeuronNum
        self._oNeuronNum = oNeuronNum


class DNN:

    def __init__(self, mode, input, layerSizes, lr):
    # layer_sizes is a np array of integers
    # mode : 1 -> train, 0 -> test
        self._parameter         = []
        self._intranets         = []
        self._layerSizes        = layerSizes
        self._intranetNum       = len(layerSizes) - 1
        self._lr                = lr
        print ('lr =', lr)
        self._output = T.matrix(dtype='float32')
        self._input = input
        self._mode = mode

        # initialize first and hidden intranets
        for i in range(0, self._intranetNum-1):
            self._intranets.append(
                InterNetwork(
                    iNeuronNum = self._layerSizes[i],
                    oNeuronNum = self._layerSizes[i+1],
                    activateType = 'ReLU',
                )
            )
            self._parameter.extend(self._intranets[i]._parameter)

        self._intranets.append(
                InterNetwork(
                    iNeuronNum = self._layerSizes[self._intranetNum-1],
                    oNeuronNum = self._layerSizes[self._intranetNum],
                    activateType = 'SoftMax',
                )
        )
        self._parameter.extend(self._intranets[self._intranetNum-1]._parameter)
        self._movement = np.ones(len(self._parameter)).astype(dtype='float32') / 1E8
        self._rng = np.random.RandomState(1234)
        

    def update(self, gParameter, eta) :
    # update parameter set , movement set
        self._movement = [(eta * v - self._lr * gp).astype(dtype='float32') for v, gp in zip(self._movement, gParameter)]
        print ([v for v in self._movement])
        # print (self._movement[0].type.dtype)
        # print ((eta * v - self._lr * gp).type.dtype for v, gp in zip(self._movement, gParameter))

        return [ (p,p + v ) # Why / rmgp is wrong ?????
                for p, v in zip(self._parameter, self._movement) ]

    def forward(self, input, intranet) :
        intranet._output = T.transpose(T.dot(intranet._weight, T.transpose(input))+ intranet._bias.dimshuffle(0,'x'))

    def activate(self, intranet) :
        if intranet._activateType == 'ReLU':
            # pass
            intranet._output = T.switch(intranet._output < 0, 0, intranet._output)
            # dropout

        elif intranet._activateType == 'SoftMax':
            MAX = T.max(intranet._output,1)
            intranet._output = intranet._output - MAX.dimshuffle(0,'x')
            intranet._output = T.exp(intranet._output)
            intranet._output = intranet._output/T.sum(intranet._output,1).dimshuffle(0,'x')
        else:
            raise ValueError

    def dropout(self, intranet, rng) :
        if intranet._activateType == 'ReLU':
            srng = T.shared_randomstreams.RandomStreams(seed = rng.randint(2 ** 30))
            if self._mode == 1:
                intranet._output = intranet._output * srng.binomial(size = (intranet._oNeuronNum), p = 0.5, dtype='float32')
            else:
                intranet._output = intranet._output * 0.5

    def feedforward(self) :
        #forward first intranet
        self.forward(self._input,self._intranets[0])
        self.activate(self._intranets[0])
        self.dropout(self._intranets[0], self._rng)

        #forward hidden intranet
        for i in range(1, self._intranetNum-1) :
            self.forward(self._intranets[i-1]._output,self._intranets[i])
            self.activate(self._intranets[i])
            self.dropout(self._intranets[i], self._rng)

        #forward last intranet
        self.forward(self._intranets[self._intranetNum-2]._output,self._intranets[self._intranetNum-1])
        self.activate(self._intranets[self._intranetNum-1])
        self.dropout(self._intranets[self._intranetNum-1], self._rng)

        self._output = self._intranets[self._intranetNum-1]._output

    def costGenerate(self, labelFeature, batchSize) :
        cost = -T.log( self._output[0][T.argmax(labelFeature[0])] )
        for i in range(1, batchSize):
            cost += -T.log( self._output[i][T.argmax(labelFeature[i])] )
        return cost

    def calculateGrad(self, cost) :
        return T.grad(cost, self._parameter)

    

    
