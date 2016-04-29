
from os import listdir
from os.path import isfile, join

from mail_corpus import MailCorpus

class TRECCorpus(MailCorpus):

    def getFilesIterator(self, directory):
        numMails = len(listdir(directory))
        return [join(directory, 'inmail.%d' % i) for i in xrange(1, numMails + 1)]

    def processMail(self, fullText):
        headersBodyDivider = fullText.find("\n\n")
        subjectIndex = fullText.find('Subject: ') + len('Subject: ')
        subjectEnd = fullText[subjectIndex + 1:].find("\n") + subjectIndex + 1

        output = {}
        output['headers'] = fullText[:headersBodyDivider]
        output['subject'] = fullText[subjectIndex:subjectEnd]
        output['body'] = fullText[headersBodyDivider + 1:]

        return output
