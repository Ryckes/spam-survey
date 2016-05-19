
import os
import itertools
import cPickle as pickle

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

random_seed = 1

training_corpus = CommonCorpus('./processed_corpora/TREC2007_untokenized')
test_corpus = CommonCorpus('./processed_corpora/LingSpam_untokenized')
validation_corpus = test_corpus


validation_size = 200
test_size = 1000

training_corpus.shuffle(random_seed)
# validation_corpus.shuffle(random_seed)
test_corpus.shuffle(random_seed)

training_data_generator = training_corpus.getMails
validation_data_generator = validation_corpus.getMails
test_data_generator = test_corpus.getMails

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

feature_maxima_file = './processed_corpora/TREC+Ling all additional features maxima'
if not os.path.exists(feature_maxima_file):
    allmails = itertools.chain(training_corpus.getMails(),
                               test_corpus.getMails())
    additional_features_maxima = util.getFeaturesMaxima(allmails,
                                                        additional_features)
    with open(feature_maxima_file, 'w') as handler:
        pickle.dump(additional_features_maxima, handler)
else:
    with open(feature_maxima_file, 'r') as handler:
        additional_features_maxima = pickle.load(handler)

n_features = len(additional_features) + vocabulary_size
batch_size = 500

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
outputDir = os.path.join(experimentsFolder, 'Different corpora 2 bigger bs')

classes = [0, 1]

training_data_iterator = DataIterator(training_data_generator,
                                      batch_size, 0,
                                      additional_features, vectorizer,
                                      additional_features_maxima)

validation_data = DataIterator(validation_data_generator,
                               validation_size, 0,
                               additional_features, vectorizer,
                               additional_features_maxima).next()
validation_measure = (util.getMeasureAverage(util.getSpecificity,
                                             util.getAccuracy),
                      'Accuracy and specificity average')

test_data = DataIterator(test_data_generator, test_size, validation_size,
                         additional_features, vectorizer,
                         additional_features_maxima).next()

test_measures = [(util.getSpecificity, 'Specificity'),
                 (util.getRecall, 'Recall'),
                 (util.getAccuracy, 'Accuracy')]


runner = SpamExperimentRunner(zip(classifiers, classifierNames),
                              training_data_iterator, classes,
                              vectorizer, outputDir, test_data,
                              test_measures, validation_data,
                              validation_measure)

runner.run()
