import IO
import rnnTrainFunc

###
#set initial train circumstance

inputDataTrain, keyOrderTrain = IO.readFile('trainPossibility.txt')
label = IO.readLabel('data/label/train.lab', 48)
# trainFunc.batchSize = 1
inputBatchesTr, labelBatchesTr = trainFunc.makeBatch(inputDataTrain, keyOrderTrain, label, 'train')

###
#start training
trainFunc.training(10, inputBatchesTr, labelBatchesTr)

#set initial test circumstance
inputDataTest, keyOrderTest = IO.readFile('testPossibility.txt')
inputBatchesTest, nothing= trainFunc.makeBatch(inputDataTest, keyOrderTest, [], 'test')
###
#start testing
outputDataTest, possibilityVectors = trainFunc.testing(inputBatchesTest, keyOrderTest)

IO.writeFile('solution.csv', 'useless', possibilityVectors, outputDataTest, keyOrderTest, 'rnn')
