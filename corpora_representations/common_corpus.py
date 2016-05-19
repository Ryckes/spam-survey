
import cPickle as pickle
from os import listdir
from os.path import join

from corpora_representations.mail_corpus import MailCorpus

class CommonCorpus(MailCorpus):
    def getFilesList(self):
        numMails = len(listdir(self.directory))
        return [join(self.directory, '%d' % i)
                for i in xrange(1, numMails + 1)]

    def processMail(self, filename, text):
        return pickle.loads(text)
