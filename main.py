import IO
import trainFunc

###
#set initial train circumstance

inputDataTrain, keyOrderTrain = IO.readFile('data/mfcc/try1.ark', 'dnn')
label = IO.readLabel('data/label/try1.lab', 48, 'dnn')
# trainFunc.batchSize = 1
inputBatchesTr, labelBatchesTr = trainFunc.makeBatch(inputDataTrain, keyOrderTrain, label, 'train')

###
#start training
trainFunc.training(100, inputBatchesTr, labelBatchesTr)

#set initial test circumstance
inputDataTest, keyOrderTest = IO.readFile('data/mfcc/try1.ark', 'dnn')
inputBatchesTest, nothing= trainFunc.makeBatch(inputDataTest, keyOrderTest, [], 'test')
###
#start testing
outputDataTest, possibilityVectors = trainFunc.testing(inputBatchesTest, keyOrderTest)

IO.writeFile('predict.csv', 'possibility.txt', possibilityVectors, outputDataTest, keyOrderTest)
