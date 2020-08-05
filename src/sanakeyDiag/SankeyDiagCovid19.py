import argparse
import csv
import numpy as np
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import pickle
import pandas as pd
# import tensorflow as tf
import json
import sklearn
from sklearn.model_selection import train_test_split
import math

import warnings

import param4BN_learn_from_data_tranfs
import utilities
from DataPreProcessing import DataPreProcessing
from DataPreProcessingUniversal import DataPreProcessing4GP00, DataPreProcessingVarModelsGrp1
from DataPreprocessingContext4Covid import DataPreprocessingContext4Covid
from param4BN_learn_from_data_tranfs import readData

warnings.filterwarnings('ignore')

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.multiclass import unique_labels
import DataPreProcessing as dp
import pathlib
from sklearn.preprocessing import KBinsDiscretizer

dateStampStr = utilities.getDateTimeStamp2Str()

class DataPreprocessingCovid4SankeyV00(DataPreProcessing):
    """
    Age and ITU admissions for SAnkey diagram
    """

    def cleanDF(self, df, colName="STUDY_ID"):
        df = df[df[colName].notna()]
        return df

    @property
    def nRows2Read(self):
        pass

    @property
    def processingVersion(self):
        pass

    def preprocess(self):
        # Demographics
        self._df_demographics = self.preProcessDemogData(self._df_demographics)
        # Events
        self._df_events = self.preProcessEvents()
        # Joined
        df = self._df_demographics.merge(self._df_events, on='STUDY_ID', how="inner")
        dfs4SaDiag: dict = self.df4SankeyD(df)
        return dfs4SaDiag

    def preProcessDemogData(self, df):
        selColsDemog = ['STUDY_ID', 'PATIENT_AGE']
        df = df[selColsDemog]
        df = self.discretizeAge(df)
        # Index(['STUDY_ID', 'PATIENT_AGE', 'AGE_CATEG', 'AGE_RANGE'], dtype='object')
        selColsDemog = ['STUDY_ID', 'PATIENT_AGE', 'AGE_CATEG', 'AGE_RANGE']
        df = df[selColsDemog]
        self._df_demographics = df
        return df

    def preprocessAndSave(self, dataFrame):
        pass

    def saveToCSVFile(self, resultingDataFrames, nRows2Read=-1, inputFileInfoStr=""):
        grpSaDiagByAge:pd.DataFrame = resultingDataFrames.get('grpSaDiagByAge')
        grpSaDiagByAgeITU = resultingDataFrames.get('grpSaDiagByAgeITU')
        processingVersion = self._processingVersion
        outputPath = self._outputPath
        nRows2Read = self.nRows2Read
        grpSaDiagByAgePath = outputPath/(f'grpSaDiagByAge_{processingVersion}_{dateStampStr}.csv')
        grpSaDiagByAgeITUPath = outputPath/(f'grpSaDiagByAgeITUe_{processingVersion}_{dateStampStr}.csv')
        #  **************************************************************
        grpSaDiagByAge.to_csv(grpSaDiagByAgePath,index=False)
        grpSaDiagByAgeITU.to_csv(grpSaDiagByAgeITUPath,index=False)
        print(f"1st Output written into file: {grpSaDiagByAgePath} ")
        print(f"2nd Output written into file: {grpSaDiagByAgeITUPath} ")
        pass

    @staticmethod
    def readData(dataFileN, nRows2REad):
        df1 = pd.DataFrame()
        if nRows2REad <= 0:
            df1 = pd.read_csv(dataFileN)
        else:
            df1 = pd.read_csv(dataFileN, nrows=nRows2REad)
        return df1

    def __init__(self, path2DataDir, nRows2Read=-1):
        self._processingVersion = 'Ph2_Covid_Sankey'
        dirPath = pathlib.Path(path2DataDir)
        # =================================
        # Demograhic data
        # =================================
        self._demographicsFilePath = dirPath / ('REACT_Demographics.csv')
        parentDir2 = self.extractParentDir(self._demographicsFilePath)
        self._outputPath = self.getOutputPath(parentDir2)
        self._df_demographics: pd.DataFrame = self.readData(self._demographicsFilePath, nRows2Read)
        # self.processDemog()
        # =================================
        # Event DAta
        # =================================
        self._eventsFilePath = path2DataDir + 'REACT_Events' + '.csv'
        self._df_events: pd.DataFrame = param4BN_learn_from_data_tranfs.readData(self._eventsFilePath, nRows2Read)

        # ===========================
        nRow2Read = -1
        # ===========================
        self.oAgeColName = "PATIENT_AGE"
        # self.extractInfoFromInputFileName(self._demographicsFilePath)

    def preProcessEvents(self):
        self._df_events = self.cleanDF(self._df_events)
        selColsEvents = ['END_DATETIME',
                         'EVENT_TYPE',
                         'START_DATETIME',
                         'STUDY_ID']

        df_event: pd.DataFrame = self._df_events[selColsEvents]
        df_event['HAS_ITU'] = (df_event.EVENT_TYPE == 'ITU')
        self._df_events = df_event
        # Index(['END_DATETIME', 'EVENT_TYPE', 'START_DATETIME', 'STUDY_ID', 'HAS_ITU'], dtype='object')
        x = df_event.groupby(by=['STUDY_ID']).agg({'HAS_ITU': lambda x: x.any()})
        x.reset_index(inplace=True)
        nITUs = len(x[x['HAS_ITU'] == True])
        nTotal = len(x)
        self._df_events = x
        return x

    def getOutputPath(self, parentDir2):
        outputPath = pathlib.Path(parentDir2) / "Output"
        pathlib.Path.mkdir(outputPath, exist_ok=True)
        return outputPath

    def extractParentDir(self, dataFileName):
        purePath = pathlib.PurePath(dataFileName)
        parentDir2 = purePath.parent.parent
        return parentDir2

    def discretizeAge(self, df):
        discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        # discretizer:KBinsDiscretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        ageVals = df[self.oAgeColName].values.reshape(-1, 1)
        discretizer.fit(ageVals)
        ageCateg = discretizer.transform(ageVals)
        # getting meaningfull lables
        df['AGE_CATEG'] = ageCateg
        df = self.newAgeLabels(discretizer, df)
        self._demographicsAgeDiscretizer = discretizer
        self._df_demographics = df
        return df

    def newAgeLabels(self, discretizer, df):
        ageBinEdges = discretizer.bin_edges_
        # ***********************************
        x = ageBinEdges[0]
        x.shape
        type(x)
        x[0]
        y = x.tolist()
        # ***********************************
        y1 = y[:-1]
        y2 = y[1:]
        yzip = zip(y1, y2)
        # ***********************************
        labelTranslation = dict()
        for idx, rr in enumerate(yzip):
            lb = rr[0]
            ub = rr[1]
            x = (lb + ub) / 2
            print(f"(idx,lb,ub) = ({idx},{rr},{rr})")
            tmp1 = np.array([x])
            print(f"tmp1 = {tmp1}")
            xReshape = np.array([x]).reshape(-1, 1)
            oldLabel = discretizer.transform(xReshape)
            ol = oldLabel[0, 0]
            newLabel = f"{lb}-{ub}"
            labelTranslation[ol] = newLabel
        df['AGE_RANGE'] = df['AGE_CATEG'].map(labelTranslation)
        self._df_demographics = df
        return df

        # Index(['STUDY_ID', 'PATIENT_AGE', 'AGE_CATEG'], dtype='object')

    def df4SankeyD(self, df):
        # grpSaDiagByAgeITU:pd.Series = df.groupby(['AGE_RANGE', 'HAS_ITU'],as_index=False).agg({'STUDY_ID': ['count']})
        grpSaDiagByAgeITU: pd.Series = df.groupby(['AGE_RANGE', 'HAS_ITU']).agg(
            COUNT=('STUDY_ID', 'count'), ).reset_index()
        # grpSaDiagByAgeITU:pd.Series = df.groupby(['AGE_RANGE', 'HAS_ITU'],as_index=False)[['STUDY_ID']].count()
        # grpSaDiagByAgeITU.rename({'STUDY_ID':'COUNT'},inplace=True)
        # grpSaDiagByAgeITU.reset_index()
        grpSaDiagByAge: pd.Series = df.groupby('AGE_RANGE').agg(COUNT=('STUDY_ID', 'count'), ).reset_index()
        # grpSaDiagByAge:pd.Series = df.groupby('AGE_RANGE',as_index=False)[['STUDY_ID']].count()
        # grpSaDiagByAge.rename({'STUDY_ID': 'COUNT'},inplace=True)
        # grpSaDiagByAge.reset_index()
        return {'grpSaDiagByAge': grpSaDiagByAge, 'grpSaDiagByAgeITU': grpSaDiagByAgeITU}


if __name__ == "__main__":
    def main():
        path2DataDir = r"C:\work\dev\dECMT_src\data_all\COVID19_Data\Current\\"
        # df_demographics: pd.DataFrame = pd.read_csv(path2DataDir + 'REACT_Demographics' + '.csv')
        # list(df_demographics.columns)
        # *************************************************************
        dataFileN = params.inputDataSetFile
        preProcessorName = params.preprocessorID
        dataProcessor = None
        nRows2REad = -1
        dataProcessor = DataPreprocessingCovid4SankeyV00(path2DataDir, nRows2Read=-1)
        # if preProcessorName == 'Ph2_GP_V00':
        #     dataProcessor = DataPreProcessing4GP00()
        # if preProcessorName == 'Ph2_VMG1_V00':  # Various models group 1
        #     dataProcessor = DataPreProcessingVarModelsGrp1()
        # if preProcessorName == ''Ph2_Covid_Sankey'':  # Various models group 1
        #     dataProcessor = DataPreProcessingVarModelsGrp1()

        dataPreprocessingContext = DataPreprocessingContext4Covid(dataProcessor)
        # dataPreprocessingContext.setDataPreProcessor(dataProcessor)
        results = dataPreprocessingContext.preprocess()
        dataPreprocessingContext.saveToCSVFile(results)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple")
    parser.add_argument("-n", action="store", dest="numRows", type=int, default=2000,
                        help="number of rows to read")

    parser.add_argument("--fds", action="store", dest="inputDataSetFile", type=str,
                        default='../../data/AKI_data_200325_full_dob_v02_test.csv',
                        help="file path to file with csv dataset file (default = '../../data/AKI_data_200325_full_dob_v02_test.csv')")

    parser.add_argument("--ppId", action="store", dest="preprocessorID", type=str, default='BNv01',
                        help="Preprocessor ID: BNv01 = for BN with excluding record if AKI present previosly in past 4 weeks")

    params = parser.parse_args()  #
    print("*****************************")
    print("Input parameters:")
    print(params)
    print("*****************************")
    main()
