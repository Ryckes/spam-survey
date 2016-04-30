
import cPickle as pickle

from common_corpus import CommonCorpus
from progress_display import ProgressDisplay

def generateDictionaries(corpus, maildir, numWords):
    histogram = {}
    files = corpus.getFilesList(maildir)
    progress = ProgressDisplay(len(files), 'Reading data')
    for filepath in files:
        with open(filepath, 'r') as fileHandler:
            mail = corpus.processMail(fileHandler.read())
            for token in mail['body']:
                try:
                    histogram[token] += 1
                except KeyError:
                    histogram[token] = 1
        progress.update()

    words = histogram.keys()
    # Descending order
    words.sort(key = lambda x: histogram[x], reverse = True)
    dictionary = dict(zip(range(numWords), words))
    inverseDictionary = dict(zip(words, range(numWords)))

    return dictionary, inverseDictionary

def main():
    corpus = CommonCorpus()
    maildir = './processed_corpora/TREC2007'
    dictOutputFile = './processed_corpora/TREC2007_dict'
    numWords = 1000
    dictionary, inverseDictionary = generateDictionaries(corpus, maildir, numWords)

    with open(dictOutputFile, 'w') as fileHandler:
        pickle.dump(dictionary, fileHandler)
    with open(dictOutputFile + '_inverse', 'w') as fileHandler:
        pickle.dump(inverseDictionary, fileHandler)

if __name__ == '__main__':
    main()
