
import os
import cPickle as pickle

from experiment_runner import ExperimentRunner

class AutostoredExperimentRunner(ExperimentRunner):
    def __init__(self, classifiers, training_data_iterator, classes,
                 vectorizer, output_directory, test_data,
                 test_measures):
        super(AutostoredExperimentRunner, self).__init__(classifiers,
                                                         training_data_iterator,
                                                         classes,
                                                         vectorizer)
        if os.path.exists(output_directory):
            raise Exception('Directory %s already exists. Please, change output directory or remove the existing one, then try again.' % output_directory)

        self.output_directory = output_directory
        self.test_data = test_data
        self.test_measures = test_measures

    def finishExperiment(self):
        # Store classifiers and results
        os.mkdir(self.output_directory)
        X, labels = self.test_data

        predictions = [classifier.predict(X) for classifier, _ in
                       self.classifiers]
        results = [{ measureName: measure(prediction, labels) for
                     measure, measureName in self.test_measures} for
                   prediction in predictions]
        for i in xrange(len(results)):
            results[i]['classifierName'] = self.classifiers[i][1]
            results[i]['classifier'] = self.classifiers[i][0]

        with open(os.path.join(self.output_directory, 'results'), 'w') as fileHandler:
            pickle.dump(results, fileHandler)
