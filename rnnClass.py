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

    def __init__(self, mode, input, layerSizes, lr):
    # layer_sizes is a np array of integers
    # mode : 1 -> train, 0 -> test
        dnnClass.DNN.__init__(self, mode, input, layerSizes, lr)
        # self._aPrev             = a_0
        self._memoryList = []
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



    def forward(self, networkInput, memory, intranet, a_tm1) :
        # input1 is the output of prev intranet
        # input2 is the memorized intranet output
            intranet._output = memory._output = T.transpose(T.dot(intranet._weight, T.transpose(networkInput)) + intranet._bias.dimshuffle(0,'x'))                                                + T.transpose(T.dot(memory._weight, T.transpose(a_tm1)))

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


    def feedforward(self, a_tm1List) :
        #forward first intranet 
        self.forward(self._input, self._memories[0], self._intranets[0], a_tm1List[0])
        self.activate(self._intranets[0])
        self._memoryList.append(self._intranets[0]._output)

        #forward hidden intranet 
        for i in range(1, self._intranetNum) :
            self.forward(self._intranets[i-1]._output, self._memories[i], self._intranets[i], a_tm1List[i])
            self.activate(self._intranets[i])
            self._memoryList.append(self._intranets[i]._output)

        # #forward last intranet 
        # # self._aPrev[0] & self._memories[0] are only useless inputs
        # self.forward(self._intranets[self._intranetNum-2]._output, self._memories[self._intranetNum-1], self._intranets[self._intranetNum-1], a_tm1List[-1])
        # self.activate(self._intranets[self._intranetNum-1])
        # self._memoryList.append(self._intranets[self._intranetNum-1]._output)

        self._output = self._intranets[self._intranetNum-1]._output

        return T.stack(self._memoryList)


    def update(self, gParameter, eta) :
    # Maintain root mean square of the gradient to update learning rate
    # update parameter set , movement set
        # self._parameter = [(p - v) for p, v in zip(self._parameter, self._movement)]
        lastMovement = self._movement
        self._movement = [(eta * v - self._lr * gp) for v, gp in zip(self._movement, gParameter)]

        return [(p, p - lv + 2*v) for p, lv, v in zip(self._parameter, lastMovement, self._movement)]

    def clipGrad(self, cost) :
        grad = T.grad(cost, self._parameter)
        clipGrad = [T.clip(g, -10, 10) for g in grad]

        return clipGrad
