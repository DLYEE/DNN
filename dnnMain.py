import IO
import mainFunc

###
#set initial train circumstance

inputDataTrain, keyOrderTrain = IO.readFile('data/mfcc/try1.ark')
label = IO.readLabel('data/label/try1.lab', 48)
# mainFunc.batchSize = 1
inputBatchesTr, labelBatchesTr = mainFunc.makeBatch(inputDataTrain, keyOrderTrain, label, 'train')

###
#start training
mainFunc.training(10, inputBatchesTr, labelBatchesTr)

#set initial test circumstance
inputDataTest, keyOrderTest = IO.readFile('data/mfcc/test.ark')
inputBatchesTest, nothing= mainFunc.makeBatch(inputDataTest, keyOrderTest, [], 'test')
###
#start testing
outputDataTest, possibilityVectors = mainFunc.testing(inputBatchesTest, keyOrderTest)

IO.writeFile('predict.csv', 'possibility.txt', possibilityVectors, outputDataTest, keyOrderTest)
