
import operator


from data_iterator import DataIterator



class ExperimentRunner(object):

    DEFAULT_COLUMN_WIDTH = 50

    def __init__(self, classifiers,
                 measures,
                 obtain_data_generator,
                 additional_features,
                 batch_size, test_size,
                 classes, vectorizer,
                 normalize_additional = True):
        self.classifiers = classifiers
        self.measures = measures
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

        self.column_width = self.DEFAULT_COLUMN_WIDTH

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

    def getMeasures(self, prediction, labels):
        return tuple([measure(prediction, labels) for measure, _ in self.measures])

    def getMeasureStrings(self, measures):
        return [', '.join(['%6.4f'] * len(measure)) % measure
                for measure in measures]

    def print_heading(self):
        tableWidth = (self.column_width + 1) * len(self.classifiers) - 1
        measuresLegendWidth = sum((2 + len(measureName) for _, measureName in self.measures)) - 2

        print
        print ' ' * ((tableWidth - measuresLegendWidth) / 2 - 1),
        print ', '.join([measureName for _, measureName in self.measures])
        print
        print ' '.join(['%' + str(self.column_width) + 's'] * len(self.classifiers)) % \
            tuple(map(operator.itemgetter(1), self.classifiers))

    def print_score(self, measures = None):
        measureStrings = self.getMeasureStrings(measures)
        print ' '.join(['%' + str(self.column_width) + 's'] * len(self.classifiers)) % \
            tuple(measureStrings)

    def mustStop(self, measures):
        # To be overriden
        return False

    def finishExperiment(self, test_data):
        # To be overriden
        pass

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

        while True:
            # X has shape (samples, n_features)
            X, y = dataIterator.next()

            # Training:
            for classifier, _ in self.classifiers:
                classifier.partial_fit(X, y, self.classes)

            predictions = [classifier.predict(test_data[0])
                           for classifier, _ in
                           self.classifiers]
            measures = [self.getMeasures(prediction, test_data[1])
                        for prediction in predictions]

            if self.mustStop(measures):
                break
        self.finishExperiment(test_data)
