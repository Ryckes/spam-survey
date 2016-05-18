
import os
import sys
import numpy

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

import util
from corpora_representations.common_corpus import CommonCorpus
from spam_experiment_runner import SpamExperimentRunner
from data_iterator import DataIterator

corpus = CommonCorpus()
directory = './processed_corpora/LingSpam_untokenized'
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

n_features = len(additional_features) + vocabulary_size
batch_size = 200
test_size = 400

classifiers = [SGDClassifier(random_state = random_seed,
                             class_weight = { 0: 1, 1: 1 }),
               SGDClassifier(random_state = random_seed,
                             class_weight = { 0: 1.1, 1: 1 }),
               SGDClassifier(random_state = random_seed,
                             class_weight = { 0: 1.2, 1: 1 }),
               SGDClassifier(random_state = random_seed,
                             class_weight = { 0: 1.3, 1: 1 })]
classifierNames = ['SGDClassifier_1', 'SGDClassifier_1.1',
                   'SGDClassifier_1.2', 'SGDClassifier_1.3']
experimentsFolder = 'experiments'
outputDir = os.path.join(experimentsFolder, 'Four_classifiers')

classes = [0, 1]

training_data_iterator = DataIterator(obtain_data_generator,
                                      batch_size, test_size,
                                      additional_features, vectorizer)
test_data = DataIterator(obtain_data_generator, test_size, 0,
                         additional_features, vectorizer).next()
validation_data = test_data

test_measures = [(util.getSpecificity, 'Specificity'),
                 (util.getRecall, 'Recall'),
                 (util.getAccuracy, 'Accuracy')]
validation_measure = (util.getMeasureAverage(util.getSpecificity,
                                             util.getAccuracy),
                      'Accuracy and specificity average')

runner = SpamExperimentRunner(zip(classifiers, classifierNames),
                              training_data_iterator, classes,
                              vectorizer, outputDir, test_data,
                              test_measures, validation_data,
                              validation_measure)

runner.run()
