import utilities as u
import gps.preproces_Covid_gps as ppC
import numpy as np
import pandas as pd
import pathlib as pl
import PO2_FO2_analysis

class ProcessingStrategy01():
    def __init__(self, dfs, dates2Convert, filesDict):
        self.dates2Convert = dates2Convert
        self.filesDict = filesDict

        self.dfs = dfs

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
    def compute_ICU_days(dict_, events_df, STUDY_ID):

        # dict_ - a dictionary corresponding to a single patient
        # events_df - a dataframe with events info directly from the csv file (datetime formating needed first)
        # STUDY_ID - patient id

        ## Function computes the number of days spent in the hospital and in ICU
        ## events_hosp_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where the EVENT_TYPE == ADMISSION
        ## ICU_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where the EVENT_TYPE == ITU
        icuDays = 0
        hopitalDays = 0
        hosp_start = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ADMISSION'), 'START_DATETIME'].values
        hosp_end = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ADMISSION'), 'END_DATETIME'].values
        try:
            hopitalDays = int((hosp_end - hosp_start) / np.timedelta64(1, 'D'))
        except:
            pass

        dict_['hosp_start'] = hosp_start

        ICU_start = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'START_DATETIME'].values
        ICU_end = events_df.loc[
            (events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'END_DATETIME'].values

        try:
            icuDays = int((ICU_end - ICU_start) / np.timedelta64(1, 'D'))
        except:
            pass
        dict_['icu_start'] = hosp_start

        dict_['ICU_Days'] = icuDays
        dict_['Hospital_Days'] = hopitalDays

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
        event_start_date_list = ['C5_start_date', 'INVASIVE VENTILATION_start_date', 'ITU_start_date', 'NIV_start_date']

        event_end_date_list = ['C5_end_date', 'INVASIVE VENTILATION_end_date', 'ITU_end_date',
                               'NIV_end_date']

        for event,eStartDate,eEndDate in zip(events_list,event_start_date_list,event_end_date_list):
            dict_[event] = 0
            if event in patient_events_list:
                dict_[event] = 1
                mapEvent = ((events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE']==event))
                eStartDateV = events_df.loc[mapEvent, 'START_DATETIME'].unique()
                eEndDateV = events_df.loc[mapEvent, 'END_DATETIME'].unique()
                dict_[eStartDate] = eStartDateV[0]
                dict_[eEndDate] = eEndDateV[0]
        return dict_

    def transformAdditionalInformation(self):
        demographics_df = self.dfs.get('demographics_df')
        events_df = self.dfs.get('events_df')

        data = {}
        for study_id in demographics_df['STUDY_ID'].unique():
            data[study_id] = dict()

            # assign demographics information to each patient
            data[study_id] = ProcessingStrategy01.assign_demographics(data[study_id], demographics_df, study_id)

            data[study_id] = ProcessingStrategy01.compute_ICU_days(data[study_id], events_df, study_id)

            data[study_id] = ProcessingStrategy01.assign_events(data[study_id], events_df, study_id)
        dfO = pd.DataFrame.from_dict(data)
        dfO = dfO.transpose()
        dfO.reset_index(level=0, inplace=True)
        dfO.rename(columns={'index': 'STUDY_ID'}, inplace=True)
        return dfO

    def process(self):
        dfs = self.dfs
        # Fixing dates
        self.dfs = u.CovidDataPreProcessing.convertColumns4Tables2Datetime(dfs, self.dates2Convert)
        df1 = self.transformAdditionalInformation()
        dfFO2P02 = self.getDataWithFO2PO2()
        dfFO2P02_summary = self.getMeasures(dfFO2P02,'')
        eventColSel = ['STUDY_ID', 'Hospital_Days', 'ICU_Days']
        # df2 = df1.loc[:, eventColSel]
        self.getRelativeDates(df1,dfFO2P02)
        dfRes = pd.merge(dfFO2P02, df1, on='STUDY_ID', how='inner')
        return dfRes


    def getDataWithFO2PO2(self):
        # ['STUDY_ID', 'DEPARTMENT', 'UNITFROM_DATETIME', 'UNITTO_DATETIME', 'PARAMETER', 'VALUE', 'RECORDED_DATETIME',
        # 'RECORDED_DATE', 'RECORDED_TIME', 'VALIDATION_DATETIME']
        # Selecting po2fo2 from vital signs
        vitalsigns_num_df = self.dfs.get('vitalsigns_num_df')
        vitalsigns_num_df.loc[:, 'PARAMETER'].unique()
        col4VSigns = ['STUDY_ID', 'RECORDED_DATETIME', 'PARAMETER', 'VALUE']
        pO2_fo2Map = vitalsigns_num_df['PARAMETER'] == 'pO2_FiO2'
        pO2_fo2Signs = vitalsigns_num_df.loc[pO2_fo2Map, col4VSigns]
        pO2_fo2Signs.loc[:, 'pO2_FiO2'] = pO2_fo2Signs.loc[:, 'VALUE']
        pO2_fo2Signs.drop(['VALUE', 'PARAMETER'], axis=1, inplace=True)
        return pO2_fo2Signs

    def getMeasures(self, df, colName):
        pass

    def getRelativeDates(self, dfEvents, dfFO2P02):
        indexColName = 'STUDY_ID'
        ids = dfEvents[indexColName].unique()
        # Index(['STUDY_ID', 'GENDER', 'ETHNIC_GROUP', 'IS_PREGNANT', 'PATIENT_AGE',
        #        'ADM_DATETIME', 'DISCHARGE_DATE', 'hosp_start', 'icu_start', 'ICU_Days',
        #        'Hospital_Days', 'C5', 'INVASIVE VENTILATION', 'ITU', 'NIV',
        #        'ITU_date_start_date', 'ITU_date_end_date', 'NIV_date_start_date',
        #        'NIV_date_end_date', 'C5_start_date', 'C5_end_date'],
        #       dtype='object')
        for id in ids:
            if dfEvents['NIV'] == 1:
                nivDate = dfEvents[dfEvents[indexColName]==id].loc[:, 'NIV_date_start_date']

            pass


def main():
    dirPath = "C:\work\dev\dECMT_src\data_all\COVID19_Data\Current"
    dates2Convert = u.timeVariables2Convert
    filesDict = u.filesDict
    dfs = u.readFiles(dirPath, filesDict)
    # =================================

    processor = ProcessingStrategy01(dfs, u.timeVariables2Convert, u.filesDict)
    df = processor.process()
    df.to_csv(pl.Path(dirPath)/f'po2_fo2_data{u.timeStampStr}.csv')

    print("Finished !!!")
    # pO2_fo2Signs.drop(['VALUE', 'PARAMETER'], axis=1, inplace=True)
    # ============================================

    # ['demographics_df', 'events_df', 'lab_results_df', 'pharmacy_data_df', 'covid_test_df', 'vitalsigns_cat_df',
    # 'vitalsigns_num_df']

    # ['END_DATETIME', 'EVENT_TYPE', 'START_DATETIME', 'STUDY_ID', 'START_DATE', 'START_TIME', 'END_DATE', 'END_TIME']
    # eventDF = ppC.ProcessingCovid4GPsContv01.add2EventsICUdays(eventDF)
    # eventDF = ppC.ProcessingCovid4GPsContv01.add2EventsHospitalDays(eventDF)

    # ['END_DATETIME', 'EVENT_TYPE', 'START_DATETIME', 'STUDY_ID', 'START_DATE', 'START_TIME', 'END_DATE', 'END_TIME',
    # 'Hospital_Days', 'ICU_Days']

    # ['STUDY_ID', 'DEPARTMENT', 'UNITFROM_DATETIME', 'UNITTO_DATETIME', 'PARAMETER', 'VALUE', 'RECORDED_DATETIME',
    # 'RECORDED_DATE', 'RECORDED_TIME', 'VALIDATION_DATETIME']
    # Selecting po2fo2 from vital signs



if __name__ == '__main__':
    main()
