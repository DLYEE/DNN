import IO
import rnnTrainFunc
import cPickle

###
#set initial train circumstance
inputBatches = []
labelBatches = []
keyOrder = []

def readTrain():
    global inputBatches, labelBatches, keyOrder
    inputData, keyOrder = IO.readPickle('data/label/train_fixed.lab','possibility.prb.train')
    label = IO.readLabel('data/label/train_fixed.lab', 48)
    inputBatches, labelBatches = rnnTrainFunc.makeBatch(inputData, keyOrder, label, 'train')
#
#set initial test circumstance
def readTest():
    global inputBatches, labelBatches, keyOrder
    inputBatches = None
    labelBatches = None
    keyOrder = None
    inputData, keyOrder = IO.readFile('data/posteriorgram/train.post')
    inputBatches, nothing= rnnTrainFunc.makeBatch(inputData, keyOrder, [], 'test')

###
#start training
def train(epochNum):
    global inputBatches, labelBatches
    rnnTrainFunc.training(epochNum, inputBatches, labelBatches)

#start testing
# def testTrainData():
    # global inputBatchesTrain, keyOrderTrain
    # outputDataTrain = {}
    # rnnTrainFunc.testing(inputBatchesTrain, keyOrderTrain, outputDataTrain)
    # IO.writeFile('trainSolution.csv', 'useless', [], outputDataTrain, keyOrderTrain, 'rnn')

def test():
    global inputBatches, keyOrder
    outputData = {}
    rnnTrainFunc.testing(inputBatches, keyOrder, outputData)
    inputBatches = None
    IO.writeFile('solution.csv', 'useless', [], outputData, keyOrder, 'rnn')

readTrain()
train(5)
readTest()
test()
