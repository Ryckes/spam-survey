
import resource

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

from common_corpus import CommonCorpus
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
additional_features = []
additional_features_max = [0] * len(additional_features)

n_features = len(additional_features) + vocabulary_size
batch_size = 500
test_size = 1423

classifiers = [SGDClassifier(random_state = random_seed),
               Perceptron(random_state = random_seed),
               MultinomialNB(alpha = 0.01),
               PassiveAggressiveClassifier(random_state = random_seed),
               MLPClassifier((20, ), random_state = random_seed)]
classifierNames = ['SGDClassifier', 'Perceptron',
                   'MultinomialNB',
                   'Passive-Aggressive',
                   'NN']
classes = [0, 1]

runner = ExperimentRunner(zip(classifiers, classifierNames),
                          obtain_data_generator,
                          additional_features,
                          batch_size, test_size,
                          classes, vectorizer)

runner.run()
