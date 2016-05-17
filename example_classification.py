
import os
import sys
import cPickle as pickle
import numpy

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

import util
from corpora_representations.common_corpus import CommonCorpus
from experiment_runner import ExperimentRunner

corpus = CommonCorpus()
directory = './processed_corpora/TREC2007_untokenized'
obtain_data_generator = (lambda corpus, directory:
                         lambda: corpus.getMails(directory))(corpus, directory)
random_seed = 1

vocabulary_size = 2 ** 18
vectorizer = HashingVectorizer(input = 'content',
                               decode_error = 'ignore',
                               n_features = vocabulary_size,
                               non_negative = True)

additional_features = ['numSentences', 'averageWordLength',
                       'ratioAlphaOverTextLength',
                       'averageSentenceLength',
                       'ratioPunctuationCharactersOverNumSentences',
                       'YuleK']
# additional_features = []
additional_features_max = [0] * len(additional_features)

n_features = len(additional_features) + vocabulary_size
batch_size = 500
test_size = 1423

classifiers = [SGDClassifier(random_state = random_seed,
                             class_weight = { 0: 1, 1: 1 })]
classifierNames = ['SGDClassifier_1']
experimentsFolder = 'experiments'
outputDir = os.path.join(experimentsFolder, classifierNames[0])
if not os.path.exists(experimentsFolder):
    os.mkdir(experimentsFolder)

if os.path.exists(outputDir):
    print """Directory %s already exists. Please, delete it or change the
classifier name.""" % outputDir
    sys.exit(1)

measures = [lambda *args:
            (util.getSpecificity(*args) + util.getAccuracy(*args)) / 2]
measureNames = ['Accuracy and specificity average']

classes = [0, 1]

class SpamExperimentRunner(ExperimentRunner):
    def __init__(self, classifiers, measures,
                 obtain_data_generator,
                 additional_features,
                 batch_size, test_size,
                 classes, vectorizer,
                 normalize_additional = True):
        super(SpamExperimentRunner, self).__init__(classifiers,
                                                   measures,
                                                   obtain_data_generator,
                                                   additional_features,
                                                   batch_size, test_size,
                                                   classes, vectorizer,
                                                   normalize_additional)

        self.bestModelSoFar = None
        self.bestMeasureSoFar = -1
        self.iterationsWithoutImprovement = 0
        self.maxIterationsWithoutImprovement = 10
        self.print_heading()

    def mustStop(self, measures):
        self.print_score(measures)
        measure = measures[0][0]
        if measure > self.bestMeasureSoFar:
            self.bestMeasureSoFar = measure
            self.bestModelSoFar = util.copyClassifierParameters(self.classifiers[0][0])
            self.iterationsWithoutImprovement = 0
        else:
            self.iterationsWithoutImprovement += 1
            if self.iterationsWithoutImprovement == \
               self.maxIterationsWithoutImprovement:
                return True

        return False

    def finishExperiment(self, test_data):
        # Store classifiers and results, return best model, etc.
        # Revert to best classifier

        classifier = self.classifiers[0][0]
        util.insertClassifierParameters(classifier, self.bestModelSoFar)

        predictions = classifier.predict(test_data[0])
        labels = test_data[1]
        result = [util.getSpecificity(predictions, labels),
                  util.getRecall(predictions, labels),
                  util.getAccuracy(predictions, labels)]

        resultLegend = ['Specificity', 'Recall', 'Accuracy']

        os.mkdir(outputDir)

        with open(outputDir + 'results', 'w') as fileHandler:
            pickle.dump(dict(zip(resultLegend, result)), fileHandler)

        with open(outputDir + 'model', 'w') as fileHandler:
            pickle.dump(self.classifiers[0][0], fileHandler)

runner = SpamExperimentRunner(zip(classifiers, classifierNames),
                              zip(measures, measureNames),
                              obtain_data_generator,
                              additional_features,
                              batch_size, test_size,
                              classes, vectorizer)

runner.run()
