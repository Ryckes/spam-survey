
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
        for key, val in results.items():
            print "%s: %f" % (key, val)
    except:
        print 'Error trying to open %s' % resultsPath
        sys.exit(1)

if __name__ == '__main__':
    dumpExperiment()

