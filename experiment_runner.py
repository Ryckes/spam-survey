
import operator

class ExperimentRunner(object):

    def __init__(self, classifiers, training_data_iterator, classes,
                 vectorizer):
        # List of (classifier, classifierName)
        self.classifiers = classifiers

        self.training_data = training_data_iterator
        self.classes = classes

        self.vectorizer = vectorizer

    def mustStop(self, measures):
        # To be overriden
        return False

    def finishExperiment(self):
        # To be overriden
        pass

    def run(self):
        dataIterator = self.training_data

        while True:
            # X has shape (samples, n_features)
            try:
                X, y = dataIterator.next()
            except StopIteration:
                print 'Data iterator was exhausted before training was finished.'
                break

            # Training:
            for classifier, _ in self.classifiers:
                if classifier is not None:
                    classifier.partial_fit(X, y, self.classes)

            if self.mustStop():
                break
        self.finishExperiment()
