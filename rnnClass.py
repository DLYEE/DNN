import dnnClass
import theano
import theano.tensor as T
import numpy as np
import random

# Need to initialize values in line 105, 111, 134, 135, 162
class MemoryNetwork(dnnClass.InterNetwork):

    def __init__(self, iNeuronNum, oNeuronNum, activateType):
        # self._input = []
        dnnClass.InterNetwork.__init__(self, iNeuronNum, oNeuronNum, activateType)
        self._weight = theano.shared(np.identity(oNeuronNum) / 100)
        self._parameter = [self._weight, self._bias]


class RNN(dnnClass.DNN):

    def __init__(self, mode, layerSizes, lr):
    # layer_sizes is a np array of integers
    # mode : 1 -> train, 0 -> test
        dnnClass.DNN.__init__(self, mode, layerSizes, lr)
        # self._aPrev             = a_0
        self._memories   = []        # memorys of intranet outputs
        # initialize memories
        for i in range(0, self._intranetNum-1):
            self._memories.append(
                MemoryNetwork(
                    iNeuronNum = self._layerSizes[i+1],
                    oNeuronNum = self._layerSizes[i+1],
                    activateType = 'ReLU',
                )
            )
            self._parameter.extend([self._memories[-1]._weight])
        self._memories.append(
            MemoryNetwork(
                iNeuronNum = self._layerSizes[self._intranetNum],
                oNeuronNum = self._layerSizes[self._intranetNum],
                activateType = 'SoftMax',
            )
        )
        self._parameter.extend([self._memories[-1]._weight])
        # No need to memorize y(the output of Rnn)
        # so the size of memories = intranetNum - 1


    def forward(self, networkInput, memory, intranet) :
            intranet._output = memory._output = T.transpose(T.dot(intranet._weight, T.transpose(networkInput)) + intranet._bias.dimshuffle(0,'x'))                                                + T.transpose(T.dot(memory._weight, memory._output))



    def step(self, input, exOutput) :
        #forward first intranet 
        self._memories[0]._output = exOutput
        self.forward(input, self._memories[0], self._intranets[0])
        self.activate(self._intranets[0])

        #forward hidden intranet 
        for i in range(1, self._intranetNum) :
            self.forward(self._intranets[i-1]._output, self._memories[i], self._intranets[i])
            self.activate(self._intranets[i])

        # #forward last intranet 
        # # self._aPrev[0] & self._memories[0] are just useless inputs
        self.forward(self._intranets[self._intranetNum-2]._output, self._memories[self._intranetNum-1], self._intranets[self._intranetNum-1])
        self.activate(self._intranets[self._intranetNum-1])

        # self._output = self._intranets[self._intranetNum-1]._output

        return self._intranets[self._intranetNum-1]._output  


    def update(self, cost) :
    # Maintain root mean square of the gradient to update learning rate
    # update parameter set , movement set
        nextStep = [(p + v) for p, v in zip(self._parameter, self._movement)]
        gParameter = T.grad(cost, nextStep)
        self._movement = [(eta * v - self._lr * gp) for v, gp in zip(self._movement, gParameter)]
        return [(p, p + v) for p, v in zip(self._parameter, self._movement)]

    def calculateGrad(self, cost) :
        grad = T.grad(cost, self._parameter)
        return T.clip(grad, -10, 10) 

