import dnnClass.py

# Need to initialize values in line 105, 111, 134, 135, 162

class RNN(DNN):

    def __init__(self, mode, input, layerSizes, lr, a_0):
    # layer_sizes is a np array of integers
    # mode : 1 -> train, 0 -> test
        DNN.__init__(self, mode, input, layerSizes, lr)
        self._aPrev             = a_0
        self._aList             = []
        self._memories          = []        # memorys of intranet outputs

        # initialize memories
        for i in range(0, self._intranetNum-1):
            self._momories.append(
                InterNetwork(
                    iNeuronNum = self._layerSizes[i],
                    oNeuronNum = self._layerSizes[i+1],
                    activateType = 'ReLU',
                )
            )
            self._parameter.extend(self._memories[i]._parameter)
        # No need to memorize y(the output of Rnn)
        # so the size of memories = intranetNum - 1

        self._rmgParameter = []


    def forward(self, input1, last, input2, memory, intranet) :
        # input1 is the output of prev intranet
        # input2 is the memorized intranet output
        intranet._output = T.transpose(T.dot(intranet._weight, T.transpose(input1))+ intranet._bias.dimshuffle(0,'x'))
        # last = 1 for last layer, last = 0 otherwise
        if last != 1:
            memory._output = T.transpose(T.dot(memory._weight, T.transpose(input2)) + memory._bias.dimshuffle(0,'x'))
            intranet._output = memory._output = intranet._output + memory._output

    def feedforward(self) :
        #forward first intranet 
        self.forward(self._input, 0, self._aPrev[0], self._memories[0], self._intranets[0])
        self.activate(self._intranets[0])
        self._aList.append(self._intranets[0]._output)
        
        #forward hidden intranet 
        for i in range(1, self._intranetNum-1) :
            self.forward(self._intranets[i-1]._output, 0, self._aPrev[i], self._memories[i], self._intranets[i])
            self.activate(self._intranets[i])
            self._aList.append(self._intranets[i]._output)

        #forward last intranet 
        self.forward(self._intranets[self._intranetNum-2]._output, 1, self._aPrev[0], self._memories[0], self._intranets[self._intranetNum-1])  # self._aPrev[0] & self._memories[0] are only useless inputs
        self.activate(self._intranets[self._intranetNum-1])
        self._aList = T.stack(self._aList)

        self._output = self._intranets[self._intranetNum-1]._output
