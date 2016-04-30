
from copy import copy

import nltk
from nltk.tokenize import word_tokenize

import itertools

def downloadNLTKIfNecessary():
    try:
        word_tokenize('')
    except LookupError:
        # Punkt models
        nltk.download('punkt')

class FeatureExtractor:

    DEFAULT_END_OF_SENTENCE_SYMBOLS = [',', '.', '!', '?', ':', ';']

    def __init__(self):
        self.endOfSentenceSymbols = self.DEFAULT_END_OF_SENTENCE_SYMBOLS

    def getSentences(self, tokens):
        sentences = [[]]

        for token in tokens:
            word = False
            for char in token:
                if char not in self.endOfSentenceSymbols:
                    # It's a word
                    word = True
                    sentences[-1].append(token)
                    break
            if not word:
                if len(sentences[-1]) > 0 or len(sentences) == 1:
                    sentences[-1].append(token)
                    sentences.append([])
                else:
                    sentences[-2].append(token)

        return filter(lambda x: len(x) > 0, sentences)

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

    def getNumPunctuationCharacters(self, tokens):
        punctuationCharacters = self.endOfSentenceSymbols
        return sum([1 if char in punctuationCharacters else 0
                    for token in tokens for char in token])

    def getYuleKMeasure(self, tokens):
        textDictionary = {}
        for token in tokens:
            try:
                textDictionary[token] += 1
            except KeyError:
                textDictionary[token] = 1

        # M1 is the number of all word forms a text consists of.
        M1 = len(textDictionary)

        # M2 is the sum of the products of each observed frequency to
        # the power of two and the number of word types observed with
        # that frequency.
        M2 = sum([(freq ** 2) * len(list(wordsWithThisFreq)) for
                  freq, wordsWithThisFreq in itertools.groupby(sorted(textDictionary.values()))])

        try:
            return 10000 * (M2 - M1) / float(M1 ** 2)
        except ZeroDivisionError:
            return 0

    def tokenize(self, string):
        string = unicode(string, 'ascii', 'replace').lower()
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

        # The following feature extraction procedure is rather
        # inefficient, but since we cannot improve its asymptotic
        # complexity, it may be worth the tradeoff for readability.
        featureDict['averageWordLength'] = self.getAverageWordLength(featureDict['body'])
        featureDict['ratioAlphaOverTextLength'] = self.getRatioAlphaOverTextLength(featureDict['body'])

        sentences = self.getSentences(featureDict['body'])

        featureDict['numSentences'] = len(sentences)
        numPunctuationChars = self.getNumPunctuationCharacters(featureDict['body'])
        try:
            featureDict['averageSentenceLength'] = len(featureDict['body']) / float(featureDict['numSentences'])
            featureDict['ratioPunctuationCharactersOverNumSentences'] = numPunctuationChars / float(featureDict['numSentences'])
        except ZeroDivisionError:
            featureDict['averageSentenceLength'] = 0
            featureDict['ratioPunctuationCharactersOverNumSentences'] = 0

        featureDict['YuleK'] = self.getYuleKMeasure(featureDict['body'])

        return featureDict
