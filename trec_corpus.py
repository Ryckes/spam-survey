
from os import listdir
from os.path import isfile, join

from mail_corpus import MailCorpus

class TRECCorpus(MailCorpus):

    def __init__(self):
        self.labels = None

    def getFilesList(self, directory):
        directory = join(directory, 'data')
        numMails = len(listdir(directory))
        return [join(directory, 'inmail.%d' % i) for i in xrange(1, numMails + 1)]

    def processMail(self, directory, filename, fullText):
        if self.labels is None:
            self.labels = self._getLabels(directory)

        headersBodyDivider = fullText.find("\n\n")
        subjectIndex = fullText.find('Subject: ') + len('Subject: ')
        subjectEnd = fullText[subjectIndex + 1:].find("\n") + subjectIndex + 1

        output = {}
        output['headers'] = fullText[:headersBodyDivider]
        output['subject'] = fullText[subjectIndex:subjectEnd]
        output['body'] = fullText[headersBodyDivider + 1:]
        lastDotPos = filename.rfind('.')
        output['label'] = self.labels[int(filename[lastDotPos + 1:]) - 1]

        return output

    def _getLabels(self, directory):
        directory = join(directory, 'full')
        labelsFile = join(directory, 'index')
        labels = []
        with open(labelsFile, 'r') as fileHandler:
            while True:
                line = fileHandler.readline()
                if len(line) == 0:
                    break

                txtLabel = line.split()[0]
                labels.append(1 if txtLabel == 'spam' else 0)

        return labels
