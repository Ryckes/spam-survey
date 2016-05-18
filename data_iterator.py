
import itertools
import numpy
import scipy

class DataIterator:

    def __init__(self, obtain_data_generator, batch_size, offset,
                 additional_features, vectorizer,
                 normalize_additional = True):
        self.obtain_data_generator = obtain_data_generator
        self.batch_size = batch_size
        self.additional_features = additional_features
        self.vectorizer = vectorizer

        # First 'offset' samples are unused by this iterator.  It's a
        # simple way to save some for validation or test.
        self.offset = offset

        self.resetMailsGenerator()

        # Note: this normalization is done over the data generator. If
        # different generators or corpora are used for
        # training/validation/testing the normalization won't be sound.
        if normalize_additional:
            self.additional_features_maxima = self._find_features_maxima(obtain_data_generator(), self.additional_features)
        else:
            self.additional_features_maxima = [1] * len(self.additional_features)

    def _find_features_maxima(self, mails, additional_features):
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

        additional_features_generator = ([mail[feature]
                                          for feature in additional_features]
                                         for mail in mails)
        return max_generator_of_lists(additional_features_generator)

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

