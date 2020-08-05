import pandas as pd
from datetime import datetime
from abc import ABC, abstractclassmethod
from pathlib import Path

# from utilities import

todayVal = datetime.today()
timeStampStr = todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")


def readData(dataFileN, nRows2REad=0):
    df1 = pd.DataFrame()
    if nRows2REad <= 0:
        df1 = pd.read_csv(dataFileN)
    else:
        df1 = pd.read_csv(dataFileN, nrows=nRows2REad)
    return df1


def getDateTimeStamp2Str():
    todayVal = datetime.today()
    timeStampStr = todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")
    return timeStampStr


def readFiles(dirPath, filesDict, ):
    """

    :param dirPath:
    :type dirPath:
    :param filesDict:
    :type filesDict:
    :return:
    :rtype:
    """
    # ****** PROCESSED *********************
    # vitalsigns_num_df = pd.read_csv(path + '\REACT_Vitalsigns_Numeric.csv')
    # vitalsigns_cat_df = pd.read_csv(path + '\REACT_Vitalsigns_Categorical.csv')
    # covid_test_df= pd.read_csv(path + '\REACT_UHSCOVIDTest_processed.csv')
    # pharmacy_data_df = pd.read_csv(path + '\REACT_PharmacyData.csv')
    # lab_results_df = pd.read_csv(path + '\REACT_LabResults.csv')
    # events_df = pd.read_csv(path + '\REACT_Events.csv')
    # demographics_df = pd.read_csv(path + '\REACT_COVID_Demographics_20200506.csv')
    res = {}
    for vName, fName in filesDict.items():
        filePath = Path(dirPath) / fName
        df = readData(filePath)
        res[vName] = df
    return res


filesDict = {'demographics_df': 'REACT_COVID_Demographics_20200506.csv',
             'events_df': 'REACT_Events.csv',
             'lab_results_df': 'REACT_LabResults.csv', 'pharmacy_data_df': 'REACT_LabResults.csv',
             'covid_test_df': 'REACT_UHSCOVIDTest_processed.csv',
             'vitalsigns_cat_df': 'REACT_Vitalsigns_Categorical.csv',
             'vitalsigns_num_df': 'REACT_Vitalsigns_Numeric.csv'}

timeVariables2Convert = {'demographics_df':
                             {('FIRST_POS_DATETIME', 'ADM_DATETIME', 'DISCHARGE_DATE'): '%d/%m/%Y %H:%M'},
                         'events_df':
                             {('START_DATETIME', 'END_DATETIME'): '%Y-%m-%d %H:%M:%S',
                              ('START_DATE', 'END_DATE'): '%d/%m/%Y'},
                         'lab_results_df':
                             {('PATHOLOGY_SPECIMEN_DATE'): '%Y-%m-%d %H:%M:%S',
                              ('SPECIMEN_DATE'): '%d/%m/%Y'}
                         }


def convertColumns2Datetime(df: object, columns: object, datetime_format: object) -> object:
    for column in columns:
        df.loc[:, column] = pd.to_datetime(df[column], format=datetime_format)
    return df

# ('FIRST_POS_DATETIME', 'ADM_DATETIME', 'DISCHARGE_DATE')
# Index(['STUDY_ID', 'PATIENT_AGE', 'DOB', 'DATE_OF_DEATH', 'DOD_DATE', 'GENDER',
#        'ETHNIC_GROUP', 'SMOKING_HISTORY', 'POSTCODE', 'IS_PREGNANT', 'HEIG',
#        'WEIG', 'BMI', 'FIRST_POS_DATE', 'FIRST_POS_DATE_R', 'FIRST_POS_TIME_R',
#        'ADMIT_DATETIME', 'ADM_DATE_R', 'ADM_TIME_R', 'DISCHARGEDATE',
#        'DISCHARGE_DATE_R', 'DISCHARGE_TIME_R', 'LOS', 'LOS_PREPOS', 'READM28',
#        'READM_DATETIME', 'READM_DATE', 'READM_TIME'],
#       dtype='object')

def convertDates(df, dateDictRules):
    for columnsTuple, value in dateDictRules.items():
        df = convertColumns2Datetime(df, columnsTuple, value)
    return df

def convertDatesAuto(df, listCols):
    for colName in listCols:
        df.loc[:, colName] = pd.to_datetime(df[colName])
    return df

def convertDatesWTableName(df, tableName):
    dateDictRules = timeVariables2Convert.get(tableName)
    for columnsTuple, value in dateDictRules.items():
        df = convertColumns2Datetime(df, columnsTuple, value)
    return df


class CovidDataPreProcessing(ABC):

    @abstractclassmethod
    def preprocess(self, dataFrame):
        pass

    @abstractclassmethod
    def preprocessAndSave(self, dataFrame, pathDir):
        pass

    @abstractclassmethod
    def saveToCSVFile(self, resultingDataFrames):
        pass

    @staticmethod
    def convertColumns2Datetime(df: object, columns: object, datetime_format: object) -> object:
        for column in columns:
            df.loc[:, column] = pd.to_datetime(df[column], format=datetime_format)
        return df

    @staticmethod
    def convertColumns4Tables2Datetime(dfs, conversionDicts):
        for tableN, df in dfs.items():
            res = conversionDicts.get(tableN)
            if res is not None:
                for colNames, format in res.items():
                    try:
                        df = CovidDataPreProcessing.convertColumns2Datetime(df, colNames, format)
                    except Exception as e:
                        print(e)
                    pass
                pass
            pass
        # for column in columns:
        #     df.loc[:, column] = pd.to_datetime(df[column], format=datetime_format)
        # return df
        return dfs
