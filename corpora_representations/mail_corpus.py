
class MailCorpus:
    def getFilesList(self, directory):
        pass

    def processMail(self, directory, filename, text):
        pass

    def getMails(self, directory):
        for filename in self.getFilesList(directory):
            with open(filename, 'r') as fileHandler:
                yield self.processMail(directory, filename, fileHandler.read())
