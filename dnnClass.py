import theano
import theano.tensor as T
import numpy as np
import random

# Need to initialize values in line 105, 111, 134, 135, 162

class InterNetwork:

    def __init__(self, iNeuronNum, oNeuronNum, activateType):
        # self._input = []
        self._weight = theano.shared((np.random.randn(oNeuronNum, iNeuronNum) / (iNeuronNum**0.5) ))
        self._bias = theano.shared(np.random.randn(oNeuronNum))
        self._activateType = activateType
        self._output = []
        # self._output = T.transpose(T.dot(self._weight, T.transpose(self._input))+self._bias.dimshuffle(0,'x'))
        self._parameter = [self._weight, self._bias]
        self._iNeuronNum = iNeuronNum
        self._oNeuronNum = oNeuronNum
        # self._train = train

class DNN:

    def __init__(self, mode, input, layerSizes, lr):
    # layer_sizes is a np array of integers
    # mode : 1 -> train, 0 -> test
        self._input = input
        self._parameter         = []
        self._intranets         = []
        self._layerSizes        = layerSizes
        self._intranetNum       = len(layerSizes) - 1
        self._lr                = lr
        self._mode = mode
        self._output = T.matrix('''dtype='float32' ''')
        # self._output =

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
        self._rmgParameter = []
        self._rng = np.random.RandomState(1234)


    def update(self, gParameter) :
    # Maintain root mean square of the gradient to update learning rate
    # update parameter set , movement set
        if len(self._rmgParameter) == 0 :
            self._rmgParameter = [T.max(gp, 1E-9) for gp in gParameter]
        else :
            self._rmgParameter = [ (rmgp ** 2 + gp ** 2) ** 0.5
                for rmgp , gp in zip(self._rmgParameter , gParameter) ]
        return [ (p,p - self._lr * gp ) # Why / rmgp is wrong ?????
                for p, rmgp, gp in zip(self._parameter, self._rmgParameter, gParameter) ]

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
                intranet._output = intranet._output * srng.binomial(size = (intranet._oNeuronNum), p = 0.5)
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
    
