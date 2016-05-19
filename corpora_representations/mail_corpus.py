
import random

class MailCorpus:
    def __init__(self, directory):
        self.directory = directory
        self.filesList = None

    def getFilesList(self):
        pass

    def processMail(self, filename, text):
        pass

    def shuffle(self, random_seed = None):
        if self.filesList is None:
            self.filesList = self.getFilesList()

        random.seed(random_seed)
        random.shuffle(self.filesList)

    def getMails(self):
        if self.filesList is None:
            self.filesList = self.getFilesList()

        for filename in self.filesList:
            with open(filename, 'r') as fileHandler:
                yield self.processMail(filename, fileHandler.read())
