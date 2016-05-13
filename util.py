
import numpy

def getSpecificity(predictions, labels):
    negatives = sum(1 - labels)
    trueNegatives = sum((1 - predictions) * (1 - labels))
    return trueNegatives / float(negatives)

def getRecall(predictions, labels):
    positives = sum(labels)
    truePositives = sum(predictions * labels)
    return truePositives / float(positives)

def getAccuracy(predictions, labels):
    return numpy.sum(numpy.equal(labels, predictions)) / float(len(labels))
