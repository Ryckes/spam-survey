
import sys
import os

class ProgressDisplay:
    def __init__(self, num, processDescription):
        self.num = num
        self.processDescription = processDescription

        self.index = 0
        self.lastString = ''
        self.barSize = self.getTerminalWidth()

    def getTerminalWidth(self):
        return int(os.popen('stty size', 'r').read().split()[1])

    def update(self):
        self.index += 1
        sys.stdout.write(len(self.lastString) * "\b")
        progress = self.index / float(self.num)

        fullTitle = '%s (%d%%)' % (self.processDescription,
                                   int(100 * progress))

        # 4: 2 brackets, one space, one colon [: ]
        usedChars = 4 + len(fullTitle)

        hashesPlusSpacesSize = self.barSize - usedChars
        hashes = int(hashesPlusSpacesSize * progress)
        self.lastString = '[%s: %s%s]' % (fullTitle,
                                          '#' * hashes,
                                          ' ' * (hashesPlusSpacesSize - hashes))
        sys.stdout.write(self.lastString)
        sys.stdout.flush()
