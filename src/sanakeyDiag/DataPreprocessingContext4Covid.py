class DataPreprocessingContext4Covid():
    def __init__(self, dataPreProcessor):
        self._dataPreProcessor = dataPreProcessor
        pass



    def extractInfoFromInputFileName(self, dataFileName, nRows2Read):
        # strNameMatches = extractPhase1Version(dataFileName)
        # self._phase1Version = strNameMatches["versionString"]
        # self._seriesString = strNameMatches["seriesStr"]
        # self._inFileAnnotaion = strNameMatches["annotation"]
        # self._dataFileName = dataFileName
        # self._nRows2Read = nRows2Read
        # inputFileVersionInfo = InputFileVersionInfo(strNameMatches["versionString"], strNameMatches["dataSetStr"],
        #                                             strNameMatches["dataVersionStr"], strNameMatches["dataVersion"],
        #                                             strNameMatches["seriesStr"], strNameMatches["seriesNumber"],
        #                                             strNameMatches["annotation"])
        # self._inputFileVersionInfo = inputFileVersionInfo
        pass

    @property
    def outputDirPath(self):
        return self._outputDirPath

    @property
    def inputFileVersionInfo(self):
        self._inputFileVersionInfo

    @property
    def dataFileName(self):
        return self._dataFileName

    @property
    def inFileAnnotaion(self):
        return self._inFileAnnotaion

    @property
    def seriesString(self):
        return self._seriesString

    @property
    def phase1Version(self):
        return self._phase1Version

    @property
    def dataPreProcessor(self):
        return self._dataPreProcessor

    def preprocess(self):
        results = self._dataPreProcessor.preprocess()
        return results

    def saveToCSVFile(self, results):
        # def saveToCSVFile(self, resultingDataFrames, nRows2Read, inputFileInfoStr):
        self._dataPreProcessor.saveToCSVFile(results,-1,"")
        pass
