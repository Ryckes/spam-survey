
import itertools
import numpy
import scipy

class DataIterator:

    def __init__(self, obtain_data_generator,
                 batch_size, test_size,
                 additional_features,
                 additional_features_maxima,
                 vectorizer):
        self.obtain_data_generator = obtain_data_generator
        self.batch_size = batch_size
        self.additional_features = additional_features
        self.vectorizer = vectorizer

        # test_size is just an offset. First test_size samples are
        # unused by this iterator.
        self.test_size = test_size

        # For normalization:
        self.additional_features_maxima = additional_features_maxima

        self.resetMailsGenerator()

    def populateFeatures(self, row, matrix, mail):
        for i, feature in enumerate(self.additional_features):
            matrix[row, i] = mail[feature] / self.additional_features_maxima[i]

    def resetMailsGenerator(self):
        self.mails = itertools.islice(self.obtain_data_generator(),
                                      self.test_size, None)

    def next(self):
        additionalFeatureMatrix = scipy.sparse.lil_matrix((self.batch_size,
                                                           len(self.additional_features)))
        bodies = []
        labels = []
        for i in xrange(self.batch_size):
            try:
                mail = self.mails.next()
            except StopIteration:
                self.resetMailsGenerator()
                mail = self.mails.next()

            bodies.append(mail['body'])
            labels.append(mail['label'])
            self.populateFeatures(i, additionalFeatureMatrix, mail)

        t = self.vectorizer.transform(bodies)
        if len(self.additional_features) == 0:
            return t, numpy.array(labels)
        return scipy.sparse.hstack([t, additionalFeatureMatrix]), \
            numpy.array(labels)

