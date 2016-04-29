
from copy import copy

import nltk
from nltk.tokenize import word_tokenize

def downloadNLTKIfNecessary():
    try:
        word_tokenize('')
    except LookupError:
        # Punkt models
        nltk.download('punkt')

class FeatureExtractor:

    def getAverageWordLength(self, tokens):
        if len(tokens) == 0:
            return 0

        self.totalLength = sum([len(token) for token in tokens])
        return self.totalLength / float(len(tokens))

    def getRatioAlphaOverTextLength(self, tokens):
        if len(tokens) == 0:
            return 0

        self.totalAlpha = sum([sum([1 if char.isalpha() else 0 for char in token]) for token in tokens])
        return self.totalAlpha / float(sum([len(token) for token in tokens]))

    def tokenize(self, string):
        string = unicode(string, 'ascii', 'replace')
        return word_tokenize(string)

    def process(self, mail):
        """
        mail: must be a dictionary with entries for 'headers',
        'subject', and 'body'. Subclasses of MailCorpus provide this
        kind of dictionaries. All three elements are strings.
        """

        downloadNLTKIfNecessary()


        featureDict = copy(mail)

        featureDict['headers'] = self.tokenize(featureDict['headers'])
        featureDict['subject'] = self.tokenize(featureDict['subject'])
        featureDict['body'] = self.tokenize(featureDict['body'])

        featureDict['averageWordLength'] = self.getAverageWordLength(featureDict['body'])
        featureDict['ratioAlphaOverTextLength'] = self.getRatioAlphaOverTextLength(featureDict['body'])

        return featureDict
