
import operator

import util
from autostored_experiment_runner import AutostoredExperimentRunner

class SpamExperimentRunner(AutostoredExperimentRunner):

    DEFAULT_COLUMN_WIDTH = 20

    def __init__(self, classifiers, training_data_iterator, classes,
                 vectorizer, output_directory, test_data,
                 test_measures, validation_data, validation_measure):
        super(SpamExperimentRunner, self).__init__(classifiers,
                                                   training_data_iterator,
                                                   classes,
                                                   vectorizer,
                                                   output_directory,
                                                   test_data,
                                                   test_measures)

        self.validation_data = validation_data
        self.validation_measure = validation_measure

        self.bestModelsSoFar = [None] * len(self.classifiers)
        self.bestMeasuresSoFar = [-1] * len(self.classifiers)
        self.iterationsWithoutImprovement = [0] * len(self.classifiers)
        self.maxIterationsWithoutImprovement = 10
        self.toppedClassifiers = [None] * len(self.classifiers)
        self.numClassifiersFinished = 0
        self.column_width = self.DEFAULT_COLUMN_WIDTH

        self.print_heading(validation_measure)

    def mustStop(self):
        predictions = [classifier.predict(self.validation_data[0]) if
                       classifier is not None else None for
                       classifier, _ in self.classifiers]
        measureResults = [self.validation_measure[0](prediction,
                                                     self.validation_data[1])
                          if prediction is
                          not None else None
                          for prediction in
                          predictions]

        self.print_score([(result,) for result in measureResults])

        for i in xrange(len(self.classifiers)):
            if self.classifiers[i][0] is None:
                continue

            classifier, _ = self.classifiers[i]
            measure = measureResults[i]

            if measure > self.bestMeasuresSoFar[i]:
                self.bestMeasuresSoFar[i] = measure
                self.bestModelsSoFar[i] = util.copyClassifierParameters(classifier)
                self.iterationsWithoutImprovement[i] = 0
            else:
                self.iterationsWithoutImprovement[i] += 1
                if self.iterationsWithoutImprovement[i] == \
                   self.maxIterationsWithoutImprovement:
                    self.numClassifiersFinished += 1
                    self.toppedClassifiers[i] = self.classifiers[i]
                    self.classifiers[i] = (None, None)

        return self.numClassifiersFinished == len(self.classifiers)

    def finishExperiment(self):
        # Revert to best classifiers
        for i in xrange(len(self.classifiers)):
            self.classifiers[i] = self.toppedClassifiers[i]
            util.insertClassifierParameters(self.classifiers[i][0], self.bestModelsSoFar[i])

        super(SpamExperimentRunner, self).finishExperiment()

    def getMeasures(self, measures, prediction, labels):
        return tuple([measure(prediction, labels) for measure, _ in measures])

    def getMeasureStrings(self, measures):
        length = 6 * (len(measures[0]) + 2) - 2
        pattern = ', '.join(['%6.4f'] * len(measures[0]))
        nullPlaceholder = '%6s' % '-'

        return [pattern % measure if measure[0] is not None else
                nullPlaceholder for measure in measures]

    def print_heading(self, measure):
        tableWidth = (self.column_width + 1) * len(self.classifiers) - 1
        measureLegendWidth = len(measure[1])

        print
        print ' ' * ((tableWidth - measureLegendWidth) / 2 - 1),
        print measure[1]
        print
        print ' '.join(['%' + str(self.column_width) + 's'] *
                       len(self.classifiers)) % \
            tuple(map(operator.itemgetter(1), self.classifiers))

    def print_score(self, measures):
        measureStrings = self.getMeasureStrings(measures)
        print ' '.join(['%' + str(self.column_width) + 's'] * len(self.classifiers)) % \
            tuple(measureStrings)
