
import cPickle as pickle

class MailStorage:

    def __init__(self, corpusName):
        self.corpusName = corpusName

    def store(self, mailData, name):
        with open('./processed_corpora/%s/%s' %
                  (self.corpusName, name), 'w') as fileHandler:
            pickle.dump(mailData, fileHandler)
