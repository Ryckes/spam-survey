
import itertools
import numpy
import scipy

class DataIterator:

    def __init__(self, obtain_data_generator, batch_size, offset,
                 additional_features, vectorizer,
                 additional_features_maxima = None):
        self.obtain_data_generator = obtain_data_generator
        self.batch_size = batch_size
        self.additional_features = additional_features
        self.vectorizer = vectorizer

        # First 'offset' samples are unused by this iterator.  It's a
        # simple way to save some for validation or test.
        self.offset = offset

        self.resetMailsGenerator()

        if additional_features_maxima is not None:
            self.additional_features_maxima = additional_features_maxima
        else:
            self.additional_features_maxima = [1] * len(self.additional_features)

    def populateFeatures(self, row, matrix, mail):
        for i, feature in enumerate(self.additional_features):
            matrix[row, i] = mail[feature] / self.additional_features_maxima[i]

    def resetMailsGenerator(self):
        self.mails = itertools.islice(self.obtain_data_generator(),
                                      self.offset, None)

    def reset(self):
        self.resetMailsGenerator()

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

