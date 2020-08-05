from abc import ABC, abstractmethod
from pathlib import Path
import copy

import json
import pandas as pd
import numpy as np
import seaborn as sns

import utilities as u
import utilitiesExploratoryA
from utilities import timeVariables2Convert #timeVariables2Convert
from utilitiesExploratoryA import timeStampStr, readData, list_diff
import matplotlib.pyplot as plt


def convert_columns_to_datetime(df, columns, datetime_format):
    for column in columns:
        df[column] = pd.to_datetime(df[column], format=datetime_format)

    return df


class ExploratoryAnalysisBase(ABC):

    @abstractmethod
    def analyse(self):
        pass

    @property
    def versionOfAnalysis(self):
        self._versionOfAnalysis


class ExploratoryAnalysisCovid00(ExploratoryAnalysisBase):
    """

    """

    def analyse(self,savePlots=False):
        # ['table', 'variable_name', 'nUniqueValsPC', 'estNUniqueValsPC', 'categorical']
        # dfStats = pd.DataFrame(columns=['table', 'variable_name', 'nUniqueValsPC', 'estNUniqueValsPC', 'categorical'])
        dfStats = pd.DataFrame(columns=['table', 'variable_name',  'unique_values','unique_values_counts','num_unique_vals_per_cent', 'fraction_unique_per_cent','estfraction_unique_per_cent', 'categorical', 'num_unique', 'least_frequent', 'most_frequent','freq_of_least_frequent', 'freq_of_most_frequent', 'num_of_missing_values','fraction_of_missing_values','type'])
        for tableN in self._tableNames:
            filePath = Path(self._dirPath) / f"{tableN}"
            nRows2Read = self._tableNames.get(tableN).get('nRowsToRead')
            df = readData(filePath, nRows2Read)
            res = self.analyseTable(df, tableN,savePlots)
            addDF = pd.DataFrame(res)
            dfStats = dfStats.append(addDF,ignore_index=True)
            # dfStats.append(res, ignore_index=True)
        statsFn=Path(self._outDirPath)/ f"covid_all_stats_{timeStampStr}.csv"
        dfStats.to_csv(statsFn)
        print(f"Statistics saved into : {statsFn}")


    def __init__(self, dirPath, tables: dict, timeVars2Convert):
        """
        Args:

        Returns:

        """
        self._versionOfAnalysis = "Covid_Exploratory_v00"
        self._dirPath = dirPath
        self._tableNames = tables
        parentDir2 = parentDir2 = utilitiesExploratoryA.extractParentDir(dirPath)
        outputPath = Path(parentDir2) / "Output_Covid19_Analysis"/timeStampStr
        print(f"Saving output file into directory {outputPath}")
        Path.mkdir(outputPath, exist_ok=True)
        self._outDirPath = outputPath
        self._timeVars2Convert = timeVars2Convert

    def analyseTable(self, df, tableName,savePlots):
        if savePlots:
            missingValsPath2OutputFile = Path(self._outDirPath) / f"Missing_{tableName}_missing_values_td_{timeStampStr}.png"
            pdf_missingValsPath2OutputFile = Path(self._outDirPath) / f"Missing_{tableName}_missing_values_td_{timeStampStr}.pdf"
            fig = df.isnull().mean().plot.bar() # figsize=(12,6)
            fig.axhline(y=0.05, color = 'red')
            plt.ylabel(f'Fraction of missing values')
            plt.xlabel('Variables')
            plt.title(f'Fraction of missing data  in {tableName}',pad=20)
            plt.savefig(pdf_missingValsPath2OutputFile, bbox_inches='tight', dpi=100)
            plt.savefig(missingValsPath2OutputFile, bbox_inches='tight', dpi=100)
            print(f'Writing figures  to {missingValsPath2OutputFile}')
        # plt.show()

        # *************************************************
        # ***                                        *******
        # *************************************************

        tableStats = []
        # Variable loop
        cols = df.columns
        # Index(['STUDY_ID', 'PATIENT_AGE', 'GENDER', 'ETHNIC_GROUP', 'POSTCODE',
        #        'IS_PREGNANT', 'HEIG', 'WEIG', 'BMI', 'FIRST_POS_DATE',
        #        'FIRST_POS_DATE_R', 'FIRST_POS_TIME_R', 'ADMIT_DATETIME', 'ADM_DATE_R',
        #        'ADM_TIME_R', 'DISCHARGEDATE', 'DISCHARGE_DATE_R', 'DISCHARGE_TIME_R',
        #        'LOS', 'LOS_PREPOS', 'READM28', 'READM_DATETIME', 'READM_DATE',
        #        'READM_TIME', 'DOBjan1st'],
        #       dtype='object')
        # boolCols = self.getBoolColNames(df)
        allCols = df.columns.tolist()
        for col in allCols:
            # ***********************************************
            varStats = {'table': tableName}
            varStats = self.analyseAllVariables(df, tableName, col, varStats)
            tableStats.append(varStats)

        boolCols = df.select_dtypes(include=[bool]).columns.tolist()
        contCols = df.select_dtypes(include=[np.float64]).columns.tolist()
        otherCols = list_diff(allCols, contCols)

        classifiedCols = copy.deepcopy(boolCols)
        classifiedCols.append(contCols)
        # otherCols = list_diff(allCols,classifiedCols)
        for col in allCols:
            self.analyseAllVariables(df, tableName, col, varStats,savePlots)
            # self.plotVariable(df, tableName, col, varStats)
        return tableStats

    def getBoolColNames(self, df):
        df.info()
        df.describe()
        boolCols = df.select_dtypes(include=[bool]).columns.tolist()
        return boolCols

    def analyseAllVariables(self, df: pd.DataFrame, tableName, col, rowStats,savePlots=False):
        rowStats['variable_name'] = col
        varSelected = df[col]
        rowStats['type'] = str(varSelected.dtype)
        # if varSelected.dtype == np.int64:
        #     rowStats['type'] = 'int64'
        # elif varSelected.dtype == np.float:
        #     rowStats['type'] = f'{np.float}'

        varSelected.replace(' ', np.nan)
        desc = varSelected.describe()
        nUnique = len(varSelected.unique())
        fracUniquePC = 100 * len(varSelected.unique()) / len(varSelected)
        estNUniquePC = len(varSelected[0:100].unique())
        rowStats['num_unique_vals'] = nUnique
        rowStats['fraction_unique_per_cent'] = fracUniquePC
        rowStats['estfraction_unique_per_cent'] = estNUniquePC
        rowStats['categorical'] = False
        rowStats['num_unique'] = varSelected.nunique()
        if rowStats['num_unique']<31:
            x = varSelected.value_counts().to_dict()
            xstr = json.dumps(x)
            rowStats['unique_values_counts'] = xstr
            x2 = varSelected.unique()
            x2str = np.array2string(x2)
            rowStats['unique_values'] = x2str
        else:
            rowStats['unique_values'] = ""
            rowStats['unique_values_counts'] = pd.NA
        rowStats['least_frequent'] = pd.NA
        rowStats['most_frequent'] = pd.NA
        rowStats['freq_of_least_frequent'] =pd.NA
        rowStats['freq_of_most_frequent'] = pd.NA

        if fracUniquePC < 40.0 and nUnique<30:
            rowStats['categorical'] = True
        if rowStats['categorical'] == True:
            valCounts: dict = varSelected.value_counts(ascending=True).to_dict()
            normalizedFrequncy: dict = varSelected.value_counts(normalize=True, ascending=True).to_dict()
            sortedValCounts = {k: v for k, v in sorted(valCounts.items(), key=lambda item: item[1])}
            rowStats['least_frequent'] = list(normalizedFrequncy)[0]
            rowStats['most_frequent'] = list(normalizedFrequncy)[-1]
            rowStats['freq_of_least_frequent'] = list(normalizedFrequncy.values())[0]
            rowStats['freq_of_most_frequent'] = list(normalizedFrequncy.values())[-1]
            # *** bar plot **************************
            if savePlots:
                plt.xlabel(col)
                plt.ylabel('Normalized Frequency')
                plt.title(f"Category - normalized frequency (in {tableName})", pad=20)
                plt.bar(normalizedFrequncy.keys(),normalizedFrequncy.values())
                plt.axhline(y=0.05, color='red')
                plt.subplots_adjust(bottom=0.2)
                pdfhistPath2OutputFile = Path(self._outDirPath) / f"categories_{tableName}_{col}_td_{timeStampStr}.pdf"
                histPath2OutputFile = Path(self._outDirPath) / f"categories_{tableName}_{col}_td_{timeStampStr}.png"
                print(f"File written to {histPath2OutputFile}")
                plt.savefig(histPath2OutputFile, dpi=100)
                plt.savefig(pdfhistPath2OutputFile, dpi=100)
            # plt.show()
        else:
            try:
                if savePlots:
                    path2OutputFile = Path(self._outDirPath) / f"Dist_{col}_{tableName}_dist_td_{timeStampStr}.png"
                    path2OutputFile2 = Path(self._outDirPath) / f"Dist_{col}_{tableName}_dist_td_{timeStampStr}_2.png"
                    print(f"File written to {path2OutputFile}")
                    distFig = sns.distplot(varSelected, rug=True, rug_kws={"color": "g"},
                                           kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                                           hist_kws={"histtype": "step", "linewidth": 3,
                                                     "alpha": 1, "color": "g"}).set_title(f"{col}.{tableName}")
                    distFig.get_figure().savefig(path2OutputFile)
                    plt.savefig(path2OutputFile2, dpi=100)
                # plt.show()
            except:
                pass

        # *** Missing Values
        numOfMissingValues = varSelected.isnull().sum()
        rowStats['num_of_missing_values'] = numOfMissingValues
        fractionOfMissingVals = varSelected.isnull().mean()
        rowStats['fraction_of_missing_values'] = fractionOfMissingVals
        # ****************************************************
        return rowStats

    def plotVariable(self, df, tableName, col, rowStats):
        selCol: pd.Series = df[col]
        mHist = selCol.hist(bins=20, )
        histF = mHist.get_figure()
        mHist.plot()
        histPath2OutputFile = Path(self._outDirPath) / f"Hist_var_{col}_{tableName}_hist_{timeStampStr}.png"
        histPth2OutputFile2 = Path(self._outDirPath) / f"Hist_var_{col}_{tableName}_hist_{timeStampStr}_2.png"
        pdf_histPath2OutputFile = Path(self._outDirPath) / f"Hist_var_{col}_{tableName}_hist_{timeStampStr}.pdf"
        print(f"File written to {histPath2OutputFile}")
        plt.title(f"{tableName}.{col}")
        plt.savefig(histPath2OutputFile, dpi=100)
        plt.savefig(pdf_histPath2OutputFile, dpi=100)
        histF.savefig(histPth2OutputFile2, dpi=100)
        # plt.show()
        # #============
        try:
            path2OutputFile = Path(self._outDirPath) / f"Dist_{col}_{tableName}_dist_td_{timeStampStr}_plt_var.png"
            path2OutputFile2 = Path(self._outDirPath) / f"Dist_{col}_{tableName}_dist_td_{timeStampStr}_plt_var_2.png"
            print(f"File written to {path2OutputFile}")
            distFig = sns.distplot(selCol, rug=True, rug_kws={"color": "g"},
                                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                                   hist_kws={"histtype": "step", "linewidth": 3,
                                             "alpha": 1, "color": "g"}).set_title(f"{col}.{tableName}")
            distFig.get_figure().savefig(path2OutputFile, dpi=100)
            plt.savefig(path2OutputFile2, dpi=100)
            # plt.show()
        except:
            pass
        pass


import argparse

def getListOCSVFilesInDir(data_dir,pattern):
    p = Path(data_dir).glob(pattern)
    files = [x for x in p if x.is_file()]
    return (files)


def getFileNames(filePaths):
    names = list()
    for fp in filePaths:
        names.append(Path(fp).name)
    return names

if __name__ == "__main__":

    def main(nRowsToRead=None):
        data_dir = r"C:\work\dev\dECMT_src\data_all\COVID19_Data\Current"
        patternREACT = "REACT*.csv"
        filePathsREACT = getListOCSVFilesInDir(data_dir,patternREACT)
        fileNamesREACT = getFileNames(filePathsREACT)
        nRowsToRead = -1
        tables: dict = {k:{'nRowsToRead':nRowsToRead} for k in fileNamesREACT}

        # tables: dict = {'REACT_COVID_Demographics': {'nRowsToRead': nRowsToRead},
        #                 'REACT_COVID_Demographics_20200506': {'nRowsToRead': nRowsToRead},
        #                 'REACT_Events': {'nRowsToRead': nRowsToRead},
        #                 'REACT_LabResults': {'nRowsToRead': nRowsToRead},
        #                 'REACT_PharmacyData': {'nRowsToRead': nRowsToRead},
        #                 'REACT_UHSCOVIDTest_processed': {'nRowsToRead': nRowsToRead},
        #                 'REACT_Vitalsigns_Categorical': {'nRowsToRead': nRowsToRead},
        #                 'REACT_Vitalsigns_Numeric': {'nRowsToRead': nRowsToRead}}

        explDAnalysis = ExploratoryAnalysisCovid00(data_dir, tables, u.timeVariables2Convert)
        explDAnalysis.analyse(savePlots=False)
        print("FINISHED !!!")
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
