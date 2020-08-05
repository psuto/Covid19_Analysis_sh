import utilities as u
import pandas as pd
from utilities import CovidDataPreProcessing
import collections
import numpy as np


class ProcessingCovid4GPsContv01(CovidDataPreProcessing):
    """
    Preprocessing COVID19 data for Gaussian processes (regression):
    One series and constant demographics
    Befor admition to ITU


    """

    def preprocess(self,dataFrames):
        nDict = {}
        # dict_keys(['demographics_df', 'events_df', 'lab_results_df', 'pharmacy_data_df', 'covid_test_df',
        # 'vitalsigns_cat_df', 'vitalsigns_num_df'])
        demographics_df = dataFrames.get('demographics_df')
        for studyId in demographics_df.loc[:, 'STUDY_ID'].unique():
            nDict[studyId] = self.assign_demographics()
            # nDict[studyId] =
            pass
        print()

    def preprocessAndSave(self,dataFrames):
        print()
        print()

    def saveToCSVFile(self, resultingDataFrames):
        print()

    @classmethod
    def convertDates(cls,dfs,rules):
        dateDict = rules.rules
        for key,value in dateDict.items():
            df = cls.convertColumns2Datetime(df,key,value)
            print(key, value)
        print()
        return df

    @staticmethod
    def assign_demographics(dict_, demographics_df, STUDY_ID):
        # dict_ - a dictionary corresponding to a single patient
        # demodemographics_df - a dataframe with demographics info directly from the csv file (datetime formating needed first)
        # STUDY_ID - patient id

        ## Function assignes relevent demographics information to one patient.
        ## It appends each feature name as a new 'dictinary key', and its value as 'dictionary value'.

        dict_['GENDER'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'GENDER'].values[0]
        dict_['ETHNIC_GROUP'] = \
        demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'ETHNIC_GROUP'].astype('category').values[0]
        dict_['IS_PREGNANT'] = \
        demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'IS_PREGNANT'].astype('int').values[0]
        dict_['PATIENT_AGE'] = \
        demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'PATIENT_AGE'].astype('int').values[0]
        dict_['ADM_DATETIME'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'ADM_DATETIME'].values[0]
        dict_['DISCHARGE_DATE'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'DISCHARGE_DATE'].values[
            0]

        return dict_

    @staticmethod
    def assign_events(dict_, events_df, STUDY_ID):

        # dict_ - a dictionary corresponding to a single patient
        # events_df - a dataframe with events info directly from the csv file (datetime formating needed first)
        # STUDY_ID - patient id

        ## Function assigns 0-1 values to events from event list = ['C5','INVASIVE VENTILATION','ITU','NIV']
        ## If an event takes place, the value is 1.
        ## Default value for an event is 0 (not occured)
        ## E.g. if patient has an event ITU in EVENT_TYPE column in REACT_Events.csv he gets value 1 in the dictionary with key='ITU'

        patient_events_list = list(events_df.loc[(events_df['STUDY_ID'] == STUDY_ID), 'EVENT_TYPE'].values)

        events_list = ['C5', 'INVASIVE VENTILATION', 'ITU', 'NIV']

        for event in events_list:
            dict_[event] = 0
            if event in patient_events_list:
                dict_[event] = 1

        return dict_

    @staticmethod
    def computeHospitalDays(events_df, STUDY_ID, current_date='2020-06-05'):
        """
        PS
        Function computes the number of days spent in the hospital and in ICU

        events_hosp_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where
        the EVENT_TYPE == ADMISSION

        # dict_ - a dictionary corresponding to a single patient
        # events_df - a dataframe with events info directly from the csv file (datetime formating needed first)
        # STUDY_ID - patient id
        """
        hospidaDays = 0
        hosp_start = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ADMISSION'), 'START_DATETIME'].values
        hosp_end = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ADMISSION'), 'END_DATETIME'].values
        try:
            hospidaDays = int((hosp_end - hosp_start) / np.timedelta64(1, 'D'))
        except:
            pass
        return hospidaDays

    @staticmethod
    def computeICUDays(events_df, STUDY_ID, current_date='2020-06-05'):
        """
        PS
        Function computes the number of days spent in the hospital and in ICU

        events_hosp_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where
        the EVENT_TYPE == ADMISSION

        # dict_ - a dictionary corresponding to a single patient
        # events_df - a dataframe with events info directly from the csv file (datetime formating needed first)
        # STUDY_ID - patient id
        """
        numDays = 0
        start_ = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'START_DATETIME'].values
        end_ = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'END_DATETIME'].values
        try:
            numDays = int((end_ - start_) / np.timedelta64(1, 'D'))
        except:
            pass
        return numDays

    @staticmethod
    def compute_days_in_hospital(dict_, current_date='2020-06-05'):

        # dict_ - a dictionary of given patient, with ADM_DATETIME and DISCHARGE_DATE in 'datetime64[s]' format
        # current_date - if patient has not been discharged, it calculates the number of days from ADMISSION to 'current date'

        ## function computes the number of days from ADM_DATETIME to DISCHARGE_DATE
        ## it outputs the number of days, and TRUE/FALSE if the patient has been discharged up to 'current date'

        # Has the patient been discharged?
        discharged = ~np.isnat(dict_['DISCHARGE_DATE'])

        if discharged:
            #        print('discharged')

            days_in_hospital = int(
                (dict_['DISCHARGE_DATE'] - dict_['ADM_DATETIME']).astype('timedelta64[D]') / np.timedelta64(1, 'D'))

        elif ~discharged:
            #       print('still at hospital')
            days_in_hospital = (np.array(current_date, dtype=np.datetime64) - dict_['ADM_DATETIME']).astype(
                'timedelta64[D]') / np.timedelta64(1, 'D')

        return days_in_hospital, int(discharged)


    @staticmethod
    def compute_ICU_days( events_df, STUDY_ID):
        # events_df - a dataframe with events info directly from the csv file (datetime formating needed first)
        # STUDY_ID - patient id
        ## Function computes the number of days spent in  in ICU
        ## ICU_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where the EVENT_TYPE == ITU
        icuDays = 0


        ICU_start = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'START_DATETIME'].values
        ICU_end = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'END_DATETIME'].values

        if(ICU_end is None | len(ICU_end)==0):
            pass

        try:
            icuDays = int((ICU_end - ICU_start) / np.timedelta64(1, 'D'))
        except:
            pass
        if (icuDays<=0):
            pass
        return icuDays

    @staticmethod
    def compute_ICU_days_Orig(dict_, events_df, STUDY_ID):

        # dict_ - a dictionary corresponding to a single patient
        # events_df - a dataframe with events info directly from the csv file (datetime formating needed first)
        # STUDY_ID - patient id

        ## Function computes the number of days spent in the hospital and in ICU
        ## events_hosp_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where the EVENT_TYPE == ADMISSION
        ## ICU_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where the EVENT_TYPE == ITU

        hosp_start = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ADMISSION'), 'START_DATETIME'].values
        hosp_end = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ADMISSION'), 'END_DATETIME'].values
        try:
            dict_['events_hosp_days'] = int((hosp_end - hosp_start) / np.timedelta64(1, 'D'))
        except:
            pass

        ICU_start = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'START_DATETIME'].values
        ICU_end = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'END_DATETIME'].values

        try:
            dict_['ICU_days'] = int((ICU_end - ICU_start) / np.timedelta64(1, 'D'))
        except:
            pass

        return dict_

    @staticmethod
    def add2EventsICUdays(eventDF):
        # 'STUDY_ID'
        pass
        icuDays = 0
        pass
        eventDF['ICU_Days'] = None
        #['END_DATETIME', 'EVENT_TYPE', 'START_DATETIME', 'STUDY_ID', 'START_DATE', 'START_TIME', 'END_DATE', 'END_TIME', 'ICU_Days']
        uniqueIDS = eventDF.loc[:,'STUDY_ID'].unique()
        for id in uniqueIDS:
            icuDays = ProcessingCovid4GPsContv01.computeICUDay(eventDF,id)
            eventDF.loc[eventDF['STUDY_ID']==id,'ICU_Days'] = icuDays
            return eventDF

    @staticmethod
    def add2EventsHospitalDays(eventDF):
        # 'STUDY_ID'
        pass
        hospDays = 0
        pass
        eventDF['Hospital_Days'] = None
        # ['END_DATETIME', 'EVENT_TYPE', 'START_DATETIME', 'STUDY_ID', 'START_DATE', 'START_TIME', 'END_DATE', 'END_TIME', 'ICU_Days']
        uniqueIDS = eventDF.loc[:, 'STUDY_ID'].unique()
        for id in uniqueIDS:
            hospDays = ProcessingCovid4GPsContv01.computeHospitalDays(eventDF, id)
            eventDF.loc[eventDF['STUDY_ID'] == id, 'Hospital_Days'] = hospDays
        return eventDF

    @staticmethod
    def add2EventsICUandHospitalDays(eventDF):
        # 'STUDY_ID'
        pass
        hospDays = 0
        pass
        eventDF['Hospital_Days'] = None
        eventDF['ICU_Days'] = None
        # ['END_DATETIME', 'EVENT_TYPE', 'START_DATETIME', 'STUDY_ID', 'START_DATE', 'START_TIME', 'END_DATE', 'END_TIME', 'ICU_Days']
        uniqueIDS = eventDF.loc[:, 'STUDY_ID'].unique()
        for id in uniqueIDS:
            hospDays = ProcessingCovid4GPsContv01.computeHospitalDays(eventDF, id)
            eventDF.loc[eventDF['STUDY_ID'] == id, 'Hospital_Days'] = hospDays
            icuDays = ProcessingCovid4GPsContv01.computeICUDays(eventDF, id)
            eventDF.loc[eventDF['STUDY_ID'] == id, 'ICU_Days'] = icuDays
        return eventDF

    def addDemographics(self, demographics_df, nDict):
        for studyId in demographics_df.loc[:,'STUDY_ID'].unique():
            nDict[studyId]

            pass

        pass


def main():
    dirPath = f"C:\work\dev\dECMT_src\data_all\COVID19_Data"
    # exec('a=None')
    filesDict = u.CovidDataPreProcessing.filesDict

    # eval('a'+'=None')
    dfs = u.readFiles(dirPath,filesDict)
    #  ************************************
    # demographics_df = dfs.get('demographics_df')
    # events_df = dfs.get('events_df')

    # dfM = pd.merge(demographics_df,events_df ,on='STUDY_ID',how='inner')

    # selCol = ['STUDY_ID', 'PATIENT_AGE', 'GENDER', 'ETHNIC_GROUP', 'POSTCODE', 'IS_PREGNANT', 'FIRST_POS_DATE',
    # 'FIRST_POS_DATETIME', 'FIRST_POS_TIME', 'ADM_DATE', 'ADM_DATETIME', 'ADM_TIME', 'DISCHARGE_DATE',
    # 'LOS', 'LOS_PREPOS', 'START_DATETIME', 'END_DATETIME', 'START_DATE', 'START_TIME', 'END_DATE',
    # 'END_TIME', 'EVENT_TYPE']

    # cN = list(dfM.columns)

    # cN: ['STUDY_ID', 'PATIENT_AGE', 'GENDER', 'ETHNIC_GROUP', 'POSTCODE', 'IS_PREGNANT', 'FIRST_POS_DATE',
    # 'FIRST_POS_DATETIME', 'FIRST_POS_TIME', 'ADM_DATE', 'ADM_DATETIME', 'ADM_TIME', 'DISCHARGE_DATE',
    # 'LOS', 'LOS_PREPOS', 'Unnamed: 0', 'START_DATETIME', 'END_DATETIME', 'START_DATE', 'START_TIME', 'END_DATE',
    # 'END_TIME', 'EVENT_TYPE']
    #  ************************************
    #  ===========================================================================
    # selectedCols = ['ID', 'GENDER', 'ETHNIC_GROUP', 'IS_PREGNANT', 'PATIENT_AGE']


    procCovidGPs = ProcessingCovid4GPsContv01()

    DateConversionRules = collections.namedtuple('DateConversionRules', 'rules')
    dates2Convert = {('FIRST_POS_DATETIME','ADM_DATETIME','DISCHARGE_DATE'):'%d/%m/%Y %H:%M',
                     ('FIRST_POS_DATE', 'ADM_DATE'):'%d/%m/%Y',
                     'events_df':
                         {('START_DATETIME', 'END_DATETIME'): '%Y-%m-%d %H:%M:%S',
                          ('START_DATE', 'END_DATE'): '%d/%m/%Y'},
                     'lab_results_df':
                         {('PATHOLOGY_SPECIMEN_DATE'): '%Y-%m-%d %H:%M:%S',
                          ('SPECIMEN_DATE'): '%d/%m/%Y'}
                     }

    dateConversionRules = DateConversionRules(dates2Convert)
    dfs = procCovidGPs.convertDates(dfs,dateConversionRules)
    dfs = procCovidGPs.preprocess(dfs)
    print()


    procCovidGPs.preprocessAndSave(dfs)


if __name__ == '__main__':
    main()