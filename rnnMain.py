import IO
import rnnTrainFunc

###
#set initial train circumstance

#inputDataTrain, keyOrderTrain = IO.readFile('possibility.txt')
inputDataTrain, keyOrderTrain = IO.readFile('../posteriorgram/train.post')
label = IO.readLabel('data/label/train.lab', 48)
# rnnTrainFunc.batchSize = 1
inputBatchesTr, labelBatchesTr = rnnTrainFunc.makeBatch(inputDataTrain, keyOrderTrain, label, 'train')
# print inputBatchesTr
# print labelBatchesTr

###
#start training
rnnTrainFunc.training(5, inputBatchesTr, labelBatchesTr)

#set initial test circumstance
#inputDataTest, keyOrderTest = IO.readFile('possibility.txt')
inputDataTest, keyOrderTest = IO.readFile('../posteriorgram/train.post')
inputBatchesTest, nothing= rnnTrainFunc.makeBatch(inputDataTest, keyOrderTest, [], 'test')
###
#start testing
outputDataTest, possibilityVectors = rnnTrainFunc.testing(inputBatchesTest, keyOrderTest)

IO.writeFile('solution.csv', 'useless', possibilityVectors, outputDataTest, keyOrderTest, 'rnn')
