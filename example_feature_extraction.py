
import time
import os

from corpora_representations.trec_corpus import TRECCorpus
from corpora_representations.ling_corpus import LingCorpus
from corpora_representations.mail_storage import MailStorage
from feature_extractor import FeatureExtractor
from progress_display import ProgressDisplay

def processMail(directory, filename, corpus):
    with open(filename, 'r') as fileHandler:
        fullText = fileHandler.read()
        return corpus.processMail(directory, filename, fullText)

def processDir(corpusName, mailCorpus, maildir):
    mailIterator = mailCorpus.getFilesList(maildir)
    mailStorage = MailStorage(corpusName)
    featureExtractor = FeatureExtractor()
    progress = ProgressDisplay(len(mailIterator), 'Processing emails')

    # Output files are named 1 to numMails
    index = 1
    for mail in mailIterator:
        processed = processMail(maildir, mail, mailCorpus)
        features = featureExtractor.process(processed)
        mailStorage.store(features, str(index))

        index += 1
        progress.update()

def main():
    processDir('LingSpam_untokenized',
               LingCorpus(),
               './corpora/lingspam_public')

if __name__ == '__main__':
    main()
