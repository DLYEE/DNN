import IO
import dnnMainFunc


###
#set initial train circumstance

inputDataTrain, keyOrderTrain = IO.dnnReadFile('data/fbank/try1.ark', 'data/mfcc/try1.ark')
label = IO.readLabel('data/label/try1.lab', 48)
# dnnMainFunc.batchSize = 1
inputBatchesTr, labelBatchesTr = dnnMainFunc.makeBatch(inputDataTrain, keyOrderTrain, label, 'train')

###
#start training
dnnMainFunc.training(10, inputBatchesTr, labelBatchesTr)

#set initial test circumstance
inputDataTest, keyOrderTest = IO.dnnReadFile('data/fbank/try1.ark', 'data/mfcc/try1.ark')
inputBatchesTest, nothing= dnnMainFunc.makeBatch(inputDataTest, keyOrderTest, [], 'test')
###
#start testing
outputDataTest, possibilityVectors = dnnMainFunc.testing(inputBatchesTest, keyOrderTest)
IO.writeFile('predict.csv', 'possibility.txt', possibilityVectors, outputDataTest, keyOrderTest, 'dnn')
