
import cPickle as pickle

from common_corpus import CommonCorpus
from progress_display import ProgressDisplay

def generateDictionaries(corpus, maildir, numTerms, ngramSize = 1):
    histogram = {}
    files = corpus.getFilesList(maildir)
    progress = ProgressDisplay(len(files), 'Reading data')
    if ngramSize > 1:
        nextTerm = ['$']
    else:
        nextTerm = []

    for filepath in files:
        with open(filepath, 'r') as fileHandler:
            # We know that both first two parameters are unused in
            # this corpus:
            mail = corpus.processMail(None, None, fileHandler.read())
            for token in mail['body']:
                nextTerm.append(token)
                if len(nextTerm) == ngramSize:
                    if ngramSize > 1:
                        term = tuple(nextTerm)
                    else:
                        term = nextTerm[0]
                    nextTerm = nextTerm[1:]
                else:
                    continue

                try:
                    histogram[term] += 1
                except KeyError:
                    histogram[term] = 1
        progress.update()

    terms = histogram.keys()
    # Descending order
    terms.sort(key = lambda x: histogram[x], reverse = True)
    dictionary = dict(zip(range(numTerms), terms))
    inverseDictionary = dict(zip(terms, range(numTerms)))

    return dictionary, inverseDictionary

def main():
    corpus = CommonCorpus()
    maildir = './processed_corpora/TREC2007_reduced1000'
    dictOutputFile = '%s_dict_2gram' % maildir
    numTerms = 1000
    ngramSize = 2
    dictionary, inverseDictionary = generateDictionaries(corpus, maildir, numTerms, ngramSize)

    with open(dictOutputFile, 'w') as fileHandler:
        pickle.dump(dictionary, fileHandler)
    with open(dictOutputFile + '_inverse', 'w') as fileHandler:
        pickle.dump(inverseDictionary, fileHandler)

if __name__ == '__main__':
    main()
