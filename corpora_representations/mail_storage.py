
import os
import cPickle as pickle

class MailStorage(object):

    def __init__(self, corpusName):
        self.corpusName = corpusName

    def store(self, mailData, name):
        dirname = os.path.join('./processed_corpora', self.corpusName)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print '%s created.' % dirname

        with open(os.path.join(dirname, name), 'w') as fileHandler:
            pickle.dump(mailData, fileHandler)
