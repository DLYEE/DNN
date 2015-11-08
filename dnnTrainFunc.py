import theano
import theano.tensor as T
import dnnClass
batchSize = 1

trainingMode = T.scalar(dtype='float32' )
inputDataFeature = T.matrix(dtype='float32' )
labelFeature = T.matrix(dtype='float32' )
outputDataFeature = T.vector(dtype='float32' )
neuralNetwork = dnnClass.DNN(trainingMode, inputDataFeature, [39, 128, 48], 5E-4)

neuralNetwork.feedforward()
cost = neuralNetwork.costGenerate(labelFeature, batchSize)
grad = neuralNetwork.calculateGrad(cost)

train = theano.function(
    on_unused_input = 'ignore',
    inputs = [trainingMode, inputDataFeature, labelFeature],
    updates = neuralNetwork.update(grad, 0.9),
    outputs = [cost] + [g.norm(2) for g in grad]
)

test = theano.function(
    on_unused_input = 'ignore',
    inputs = [trainingMode, inputDataFeature],
    outputs = neuralNetwork._output
)
