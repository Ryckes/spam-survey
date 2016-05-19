
import numpy

from sklearn.base import clone

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

def getMeasureAverage(*measures):
    return lambda predictions, labels: sum([measure(predictions, labels) for measure in measures]) / float(len(measures))

def copyClassifierParameters(classifier):
    param = {}
    if hasattr(classifier, 'coef_'):
        param['coef_'] = classifier.coef_.copy()
        param['intercept_'] = classifier.intercept_.copy()
    elif hasattr(classifier, 'coefs_'):
        # Mainly MLPClassifier
        param['coefs_'] = classifier.coefs_.copy()
        param['intercepts_'] = classifier.intercepts_.copy()
    else:
        raise Exception('Unrecognized classifier: %s' % classifier.__class__.__name__)

    return param

def insertClassifierParameters(classifier, parameters):
    for key, val in parameters.items():
        classifier.__dict__[key] = val

def getFeaturesMaxima(mails, features):
    def max_generator_of_lists(gen):
        prev_max = gen.next()
        try:
            # Exhaust iterator
            while True:
                el = gen.next()
                prev_max = [max(prev_max_el, current_el) for
                            prev_max_el, current_el in
                            zip(prev_max, el)]

        except StopIteration:
            pass

        return prev_max

    features_generator = ([mail[feature]
                           for feature in features]
                          for mail in mails)
    return max_generator_of_lists(features_generator)
