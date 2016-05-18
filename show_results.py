
import os
import sys
import cPickle


def dumpExperiment():
    args = sys.argv
    if len(args) != 2:
        print 'Usage: %s <experiment_path>' % args[0]
        sys.exit(1)

    path = args[1]
    resultsPath = os.path.join(path, 'results')
    try:
        with open(resultsPath, 'r') as fileHandler:
            results = cPickle.load(fileHandler)
    except:
        print 'Error trying to open %s' % resultsPath
        sys.exit(1)

    for classifierResults in results:
        for key, val in classifierResults.items():
            if key != 'classifier':
                print "%s: %s" % (key, val)
        print

if __name__ == '__main__':
    dumpExperiment()
