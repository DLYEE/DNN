import dnnClass.py

# Need to initialize values in line 105, 111, 134, 135, 162

class RNN(DNN):

    def __init__(self, mode, input, layerSizes, lr):
    # layer_sizes is a np array of integers
    # mode : 1 -> train, 0 -> test
        DNN.__init__(self, mode, input, layerSizes, lr)
        # self._aPrev             = a_0
        self._memoryList             = []
        self._memories          = []        # memorys of intranet outputs

        # initialize memories
        for i in range(0, self._intranetNum):
            self._momories.append(
                InterNetwork(
                    iNeuronNum = self._layerSizes[i+1],
                    oNeuronNum = self._layerSizes[i+1],
                    activateType = 'ReLU',
                )
            )
            self._parameter.extend(self._memories[i]._weight)
        # No need to memorize y(the output of Rnn)
        # so the size of memories = intranetNum - 1



    def forward(self, networkInput, memory, intranet) :
        # input1 is the output of prev intranet
        # input2 is the memorized intranet output
        if len(memory._output) == 0:
            intranet._output = memory._output = T.transpose(T.dot(intranet._weight, T.transpose(networkInput))+ intranet._bias.dimshuffle(0,'x'))
        else :
            intranet._output = memory._output = T.transpose(T.dot(intranet._weight, T.transpose(networkInput))+ intranet._bias.dimshuffle(0,'x')) 
                                                + T.transpose(T.dot(memory._weight, T.transpose(memory._output)) 

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
        self._memoryList = T.stack(self._memoryList)

        self._output = self._intranets[self._intranetNum-1]._output
