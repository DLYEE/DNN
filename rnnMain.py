import IO
import rnnTrainFunc

###
#set initial train circumstance
inputBatchesTrain = []
labelBatchesTrain = []
keyOrderTrain = []
def readTrain():
    global inputBatchesTrain, labelBatchesTrain, keyOrderTrain
    inputDataTrain, keyOrderTrain = IO.readFile('data/posteriorgram/train.post')
    label = IO.readLabel('data/label/train.lab', 48)
    inputBatchesTrain, labelBatchesTrain = rnnTrainFunc.makeBatch(inputDataTrain, keyOrderTrain, label, 'train')
#
#set initial test circumstance
inputBatchesTest = []
keyOrderTest = []
def readTest():
    global inputBatchesTest, keyOrderTest
    inputDataTest, keyOrderTest = IO.readFile('data/posteriorgram/train.post')
    inputBatchesTest, nothing= rnnTrainFunc.makeBatch(inputDataTest, keyOrderTest, [], 'test')

###
#start training
def train(epochNum):
    global inputBatchesTrain, labelBatchesTrain
    rnnTrainFunc.training(epochNum, inputBatchesTrain, labelBatchesTrain)

###
#start testing
# def testTrainData():
    # global inputBatchesTrain, keyOrderTrain
    # outputDataTrain = {}
    # rnnTrainFunc.testing(inputBatchesTrain, keyOrderTrain, outputDataTrain)
    # IO.writeFile('trainSolution.csv', 'useless', [], outputDataTrain, keyOrderTrain, 'rnn')

def test():
    global inputBatchesTest, keyOrderTest
    outputDataTest = {}
    rnnTrainFunc.testing(inputBatchesTest, keyOrderTest, outputDataTest)
    IO.writeFile('solution.csv', 'useless', [], outputDataTest, keyOrderTest, 'rnn')
