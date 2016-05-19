
from os import listdir
from os.path import isfile, join
import itertools

from corpora_representations.mail_corpus import MailCorpus

class LingCorpus(MailCorpus):

    def getFilesList(self, ):
        directory = join(self.directory, 'bare')
        directories = [join(directory, 'part%d' % i) for i in xrange(1, 11)]

        def filesInDirectory(directory):
            # Helper function to get files with their full path
            files = listdir(directory)
            return [join(directory, f) for f in files]

        return list(itertools.chain.from_iterable([filesInDirectory(directory) for
                                       directory in directories]))

    def processMail(self, filename, fullText):
        subjectBodyDivider = fullText.find("\n\n")

        output = {}
        output['headers'] = ''
        output['subject'] = fullText[len('Subject: '):subjectBodyDivider]
        output['body'] = fullText[subjectBodyDivider + 2:]
        # Careful: if filename contains the full or a relative path,
        # it may contain this string.
        label = 1 if 'spmsg' in filename else 0
        output['label'] = label

        return output
