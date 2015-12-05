import IO
import rnnTrainFunc
import cPickle
import hmmMain

###
#set initial train circumstance
inputBatches = []
labelBatches = []
keyOrder = []

def readTrain():
    global inputBatches, labelBatches, keyOrder
    inputBatches = None
    labelBatches = None
    keyOrder = None
    inputData, keyOrder, length = IO.readFile('../data/train.prb')
    label = IO.readLabel('../data/label/train_fixed.lab', 48)
    inputBatches, labelBatches = rnnTrainFunc.makeBatch(inputData, keyOrder, label, 'train')
#
#set initial test circumstance
def readTest():
    global inputBatches, labelBatches, keyOrder
    inputBatches = None
    labelBatches = None
    keyOrder = None
    inputData, keyOrder, length = IO.readFile('../data/test.prb')
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
    possibilityVectors = []
    rnnTrainFunc.testing(inputBatches, keyOrder, outputData, possibilityVectors)
    inputBatches = None
    IO.writeFile('../rnnFrame.csv', '../data/rnnTest.prb', possibilityVectors, outputData, keyOrder)
    IO.trimOutput('../rnnFrame.csv', '../rnn.csv')

readTrain()
train(5)
readTest()
test()
# hmmMain.hmm()
