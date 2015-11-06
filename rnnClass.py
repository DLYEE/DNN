import dnnClass
import theano
import theano.tensor as T
import numpy as np
import random

# Need to initialize values in line 105, 111, 134, 135, 162

class RNN(dnnClass.DNN):

    def __init__(self, mode, input, layerSizes, lr):
    # layer_sizes is a np array of integers
    # mode : 1 -> train, 0 -> test
        dnnClass.DNN.__init__(self, mode, input, layerSizes, lr)
        # self._aPrev             = a_0
        self._memoryList             = []
        self._memories          = []        # memorys of intranet outputs

        # initialize memories
        for i in range(0, self._intranetNum):
            self._memories.append(
                dnnClass.InterNetwork(
                    iNeuronNum = self._layerSizes[i+1],
                    oNeuronNum = self._layerSizes[i+1],
                    activateType = 'ReLU',
                )
            )
            self._memories[-1]._output = theano.shared(np.zeros(layerSizes[0]))
            self._parameter.extend(self._memories[-1]._parameter)
        # No need to memorize y(the output of Rnn)
        # so the size of memories = intranetNum - 1



    def forward(self, networkInput, memory, intranet) :
        # input1 is the output of prev intranet
        # input2 is the memorized intranet output
            intranet._output = memory._output = T.transpose(T.dot(intranet._weight, T.transpose(networkInput)) + intranet._bias.dimshuffle(0,'x'))                                                + T.transpose(T.dot(memory._weight, T.transpose(memory._output)))

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


    def feedforward(self) :
        #forward first intranet 
        self.forward(self._input, self._memories[0], self._intranets[0])
        self.activate(self._intranets[0])
        self._memoryList.append(self._intranets[0]._output)

        #forward hidden intranet 
        for i in range(1, self._intranetNum-1) :
            self.forward(self._intranets[i-1]._output, self._memories[i], self._intranets[i])
            self.activate(self._intranets[i])
            self._memoryList.append(self._intranets[i]._output)

        #forward last intranet 
        # self._aPrev[0] & self._memories[0] are only useless inputs
        self.forward(self._intranets[self._intranetNum-2]._output, self._memories[self._intranetNum-1], self._intranets[self._intranetNum-1])
        self.activate(self._intranets[self._intranetNum-1])
        self._memoryList.append(self._intranets[self._intranetNum-1]._output)

        self._memoryList = T.stack(self._memoryList)

        self._output = self._intranets[self._intranetNum-1]._output


    def update(self, gParameter) :
    # Maintain root mean square of the gradient to update learning rate
    # update parameter set , movement set
        return [ (p,p - self._lr * gp) for p, gp in zip(self._parameter, gParameter) ]

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
