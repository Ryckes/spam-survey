
from os import listdir
from os.path import isfile, join

from corpora_representations.mail_corpus import MailCorpus

class TRECCorpus(MailCorpus):

    def __init__(self, directory):
        super(TRECCorpus, self).__init__(directory)
        self.labels = None

    def getFilesList(self):
        directory = join(self.directory, 'data')
        numMails = len(listdir(self.directory))
        return [join(self.directory, 'inmail.%d' % i) for i in xrange(1, numMails + 1)]

    def processMail(self, filename, fullText):
        if self.labels is None:
            self.labels = self._getLabels()

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

    def _getLabels(self):
        directory = join(self.directory, 'full')
        labelsFile = join(self.directory, 'index')
        labels = []
        with open(labelsFile, 'r') as fileHandler:
            while True:
                line = fileHandler.readline()
                if len(line) == 0:
                    break

                txtLabel = line.split()[0]
                labels.append(1 if txtLabel == 'spam' else 0)

        return labels
