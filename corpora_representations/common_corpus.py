
import cPickle as pickle
from os import listdir
from os.path import join

from corpora_representations.mail_corpus import MailCorpus

class CommonCorpus(MailCorpus):
    def getFilesList(self, directory):
        numMails = len(listdir(directory))
        return [join(directory, '%d' % i) for i in xrange(1, numMails + 1)]

    def processMail(self, directory, filename, text):
        return pickle.loads(text)
