import IO
import rnnTrainFunc

###
#set initial train circumstance

inputDataTrain, keyOrderTrain = IO.readFile('possibility.txt')
label = IO.readLabel('data/label/train.lab', 48)
# rnnTrainFunc.batchSize = 1
inputBatchesTr, labelBatchesTr = rnnTrainFunc.makeBatch(inputDataTrain, keyOrderTrain, label, 'train')
# print inputBatchesTr
# print labelBatchesTr

###
#start training
rnnTrainFunc.training(10, inputBatchesTr, labelBatchesTr)

#set initial test circumstance
inputDataTest, keyOrderTest = IO.readFile('possibility.txt')
inputBatchesTest, nothing= rnnTrainFunc.makeBatch(inputDataTest, keyOrderTest, [], 'test')
###
#start testing
outputDataTest, possibilityVectors = rnnTrainFunc.testing(inputBatchesTest, keyOrderTest)

IO.writeFile('solution.csv', 'useless', possibilityVectors, outputDataTest, keyOrderTest, 'rnn')
