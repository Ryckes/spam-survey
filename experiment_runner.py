
import operator

from data_iterator import DataIterator

class ExperimentRunner:
    def __init__(self, classifiers, obtain_data_generator,
                 additional_features,
                 batch_size, test_size,
                 classes, vectorizer,
                 normalize_additional = True):
        self.classifiers = classifiers
        self.obtain_data_generator = obtain_data_generator
        self.additional_features = additional_features
        self.batch_size = batch_size
        self.test_size = test_size
        self.classes = classes
        self.normalize_additional = normalize_additional

        if self.normalize_additional:
            self.additional_features_maxima = self.find_features_maxima(self.obtain_data_generator(),
                                                                        self.additional_features)
        else:
            self.additional_features_maxima = [1] * len(self.additional_features)

        self.vectorizer = vectorizer

    def find_features_maxima(self, mails, additional_features):
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

    def print_heading(self):
        print ' '.join(['%20s'] * len(self.classifiers)) % \
            tuple(map(operator.itemgetter(1), self.classifiers))

    def print_score(self, classifiers, (X, y)):
        print ' '.join(['%20f'] * len(classifiers)) % \
            tuple([classifier.score(X, y)
                   for classifier, _ in
                   classifiers])

    def run(self):
        dataIterator = DataIterator(self.obtain_data_generator,
                                    self.batch_size, self.test_size,
                                    self.additional_features,
                                    self.additional_features_maxima,
                                    self.vectorizer)
        test_data = DataIterator(self.obtain_data_generator,
                                 self.test_size, 0,
                                 self.additional_features,
                                 self.additional_features_maxima,
                                 self.vectorizer).next()

        self.print_heading()
        while True:
            # X has shape (samples, n_features)
            X, y = dataIterator.next()

            for classifier, _ in self.classifiers:
                classifier.partial_fit(X, y, self.classes)
            self.print_score(self.classifiers, test_data)
