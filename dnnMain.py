import IO
import dnnMainFunc


###
#set initial train circumstance

inputDataTrain, keyOrderTrain = IO.dnnReadFile('data/fbank/train.ark', 'data/mfcc/train.ark')
label = IO.readLabel('data/label/train.lab', 48)
# dnnMainFunc.batchSize = 1
inputBatchesTr, labelBatchesTr = dnnMainFunc.makeBatch(inputDataTrain, keyOrderTrain, label, 'train')

###
#start training
dnnMainFunc.training(10, inputBatchesTr, labelBatchesTr)

#set initial test circumstance
inputDataTest, keyOrderTest = IO.readFile('data/fbank/test.ark', 'data/mfcc/test.ark')
inputBatchesTest, nothing= dnnMainFunc.makeBatch(inputDataTest, keyOrderTest, [], 'test')
###
#start testing
outputDataTest, possibilityVectors = dnnMainFunc.testing(inputBatchesTest, keyOrderTest)
IO.writeFile('predict.csv', 'possibility.txt', possibilityVectors, outputDataTest, keyOrderTest, 'dnn')
