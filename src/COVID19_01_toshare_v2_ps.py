#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[7]:


# path = r'..\..\data'
path = r'C:\work\dev\dECMT_src\data_all\COVID19_Data'


demographics_df = pd.read_csv(path + '\REACT_COVID_Demographics_20200506.csv')
events_df = pd.read_csv(path + '\REACT_Events.csv')
lab_results_df = pd.read_csv(path + '\REACT_LabResults.csv')
pharmacy_data_df = pd.read_csv(path + '\REACT_PharmacyData.csv')
covid_test_df= pd.read_csv(path + '\REACT_UHSCOVIDTest_processed.csv')
vitalsigns_cat_df = pd.read_csv(path + '\REACT_Vitalsigns_Categorical.csv')
vitalsigns_num_df = pd.read_csv(path + '\REACT_Vitalsigns_Numeric.csv')


# # How many unique patients do we have?

# In[8]:


print('REACT_COVID_Demographics_20200506 unique study_id: ', demographics_df['STUDY_ID'].unique().shape[0])
print('REACT_Events unique study_id: ', events_df['STUDY_ID'].unique().shape[0])
print('REACT_LabResults unique study_id: ', lab_results_df['STUDY_ID'].unique().shape[0])
print('REACT_PharmacyData unique study_id: ', pharmacy_data_df['STUDY_ID'].unique().shape[0])
print('REACT_UHSCOVIDTest_processed unique study_id: ', covid_test_df['STUDY_ID'].unique().shape[0])
print('REACT_Vitalsigns_Categorical unique study_id: ', vitalsigns_cat_df['STUDY_ID'].unique().shape[0])
print('REACT_Vitalsigns_Numeric unique study_id: ', vitalsigns_num_df['STUDY_ID'].unique().shape[0])


# We have all parameters for:

# In[9]:


len(set.intersection(set(demographics_df['STUDY_ID'].unique()),
                 set(events_df['STUDY_ID'].unique()),
                set(lab_results_df['STUDY_ID'].unique()),
                 set(pharmacy_data_df['STUDY_ID'].unique()),
                set(covid_test_df['STUDY_ID'].unique()),
                 set(vitalsigns_cat_df['STUDY_ID'].unique()),
                set(vitalsigns_num_df['STUDY_ID'].unique())))


# patients.

# # Create dictionary

# In[10]:


demographics_df.head()


# ## Datetime formating

# Make sure that each date time value is in proper format

# In[11]:


def convert_columns_to_datetime(df, columns, datetime_format):
    for column in columns:
        df[column] = pd.to_datetime(df[column] , format=datetime_format)

    return df


# In[12]:


demographics_df = convert_columns_to_datetime(demographics_df, columns = ['FIRST_POS_DATETIME','ADM_DATETIME','DISCHARGE_DATE'], datetime_format='%d/%m/%Y %H:%M')
demographics_df = convert_columns_to_datetime(demographics_df, columns = ['FIRST_POS_DATE', 'ADM_DATE'], datetime_format='%d/%m/%Y')


# In[13]:


events_df = convert_columns_to_datetime(events_df, columns = ['START_DATETIME','END_DATETIME'], datetime_format='%Y-%m-%d %H:%M:%S')
events_df = convert_columns_to_datetime(events_df, columns = ['START_DATE', 'END_DATE'], datetime_format='%d/%m/%Y')


# In[14]:


lab_results_df = convert_columns_to_datetime(lab_results_df, columns = ['PATHOLOGY_SPECIMEN_DATE'], datetime_format='%Y-%m-%d %H:%M:%S')
lab_results_df = convert_columns_to_datetime(lab_results_df, columns = ['SPECIMEN_DATE'], datetime_format='%d/%m/%Y')


# In[15]:


pharmacy_data_df = convert_columns_to_datetime(pharmacy_data_df, columns = ['DRUGSTARTDATE','DRUGENDDATE'], datetime_format='%Y-%m-%d %H:%M:%S')
pharmacy_data_df = convert_columns_to_datetime(pharmacy_data_df, columns = ['DRUG_STARTDATE', 'DRUG_ENDDATE'], datetime_format='%d/%m/%Y')


# In[16]:


covid_test_df = convert_columns_to_datetime(covid_test_df, columns = ['TAKEN_DATE','REPORT_DATE', 'REQUEST_DATE','ADMIT_DATETIME', 'DISCHARGEDATE'], datetime_format='%Y-%m-%d %H:%M:%S')


# In[17]:


vitalsigns_cat_df = convert_columns_to_datetime(vitalsigns_cat_df, columns = ['UNITFROM_DATETIME','UNITTO_DATETIME','RECORDED_DATETIME','VALIDATION_DATETIME'], datetime_format='%Y-%m-%d %H:%M:%S')
vitalsigns_cat_df = convert_columns_to_datetime(vitalsigns_cat_df, columns = ['RECORDED_DATE'], datetime_format='%d/%m/%Y')


# In[18]:


vitalsigns_num_df = convert_columns_to_datetime(vitalsigns_num_df, columns = ['UNITFROM_DATETIME','UNITTO_DATETIME','RECORDED_DATETIME','VALIDATION_DATETIME'], datetime_format='%Y-%m-%d %H:%M:%S')
vitalsigns_num_df = convert_columns_to_datetime(vitalsigns_num_df, columns = ['RECORDED_DATE'], datetime_format='%d/%m/%Y')


# ___

# In[19]:


def assign_demographics(dict_, demographics_df, STUDY_ID):
    
    # dict_ - a dictionary corresponding to a single patient
    # demodemographics_df - a dataframe with demographics info directly from the csv file (datetime formating needed first)
    # STUDY_ID - patient id
    
    ## Function assignes relevent demographics information to one patient.
    ## It appends each feature name as a new 'dictinary key', and its value as 'dictionary value'.
    
    dict_['GENDER'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'GENDER'].values[0]
    dict_['ETHNIC_GROUP'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'ETHNIC_GROUP'].astype('category').values[0]
    dict_['IS_PREGNANT'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'IS_PREGNANT'].astype('int').values[0]
    dict_['PATIENT_AGE'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'PATIENT_AGE'].astype('int').values[0]
    dict_['ADM_DATETIME'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'ADM_DATETIME'].values[0]
    dict_['DISCHARGE_DATE'] = demographics_df.loc[demographics_df['STUDY_ID'] == STUDY_ID, 'DISCHARGE_DATE'].values[0]
    
    return dict_


# In[20]:


'ADMISSION' in list(events_df.loc[(events_df['STUDY_ID'] == 'UHSCOVID_03b5f40f'), 'EVENT_TYPE'].values)


# In[21]:


def assign_events(dict_, events_df, STUDY_ID):
    
    # dict_ - a dictionary corresponding to a single patient
    # events_df - a dataframe with events info directly from the csv file (datetime formating needed first)
    # STUDY_ID - patient id    
    
    ## Function assigns 0-1 values to events from event list = ['C5','INVASIVE VENTILATION','ITU','NIV']
    ## If an event takes place, the value is 1. 
    ## Default value for an event is 0 (not occured)
    ## E.g. if patient has an event ITU in EVENT_TYPE column in REACT_Events.csv he gets value 1 in the dictionary with key='ITU'

    
    
    patient_events_list = list(events_df.loc[(events_df['STUDY_ID'] == STUDY_ID), 'EVENT_TYPE'].values)
    
    events_list = ['C5','INVASIVE VENTILATION','ITU','NIV']
    
    for event in events_list:
        dict_[event] = 0
        if event in patient_events_list:
            dict_[event] = 1
        
            
    return dict_


# In[22]:


def compute_ICU_days(dict_, events_df, STUDY_ID):
    
    # dict_ - a dictionary corresponding to a single patient
    # events_df - a dataframe with events info directly from the csv file (datetime formating needed first)
    # STUDY_ID - patient id    
    
    ## Function computes the number of days spent in the hospital and in ICU
    ## events_hosp_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where the EVENT_TYPE == ADMISSION
    ## ICU_days are based on REACT_Events.csv and columns START_DATETIME and END_DATETIME where the EVENT_TYPE == ITU
    
    
    hosp_start = events_df.loc[(events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ADMISSION'), 'START_DATETIME'].values
    hosp_end = events_df.loc[(events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ADMISSION'), 'END_DATETIME'].values
    try:
        dict_['events_hosp_days'] = int((hosp_end-hosp_start) / np.timedelta64(1, 'D'))
    except:
        pass
        
    ICU_start = events_df.loc[(events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'START_DATETIME'].values
    ICU_end = events_df.loc[(events_df['STUDY_ID'] == STUDY_ID) & (events_df['EVENT_TYPE'] == 'ITU'), 'END_DATETIME'].values    
        
    try:
        dict_['ICU_days'] = int((ICU_end-ICU_start) / np.timedelta64(1, 'D'))
    except:
        pass
    
    
    return dict_


# In[23]:


def compute_days_in_hospital(dict_, current_date = '2020-06-05'):
    
    # dict_ - a dictionary of given patient, with ADM_DATETIME and DISCHARGE_DATE in 'datetime64[s]' format
    # current_date - if patient has not been discharged, it calculates the number of days from ADMISSION to 'current date'
    
    ## function computes the number of days from ADM_DATETIME to DISCHARGE_DATE
    ## it outputs the number of days, and TRUE/FALSE if the patient has been discharged up to 'current date'
    
    
    # Has the patient been discharged?
    discharged = ~np.isnat(dict_['DISCHARGE_DATE'] )
    
    
    if discharged:
#        print('discharged')

        days_in_hospital = int((dict_['DISCHARGE_DATE']-dict_['ADM_DATETIME']).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
    
    elif ~discharged:
 #       print('still at hospital')
        days_in_hospital = (np.array(current_date, dtype=np.datetime64)-dict_['ADM_DATETIME']).astype('timedelta64[D]') / np.timedelta64(1, 'D')
    
    
    return days_in_hospital, int(discharged)


# In[24]:


data = {}

table_names = ['Demographics','Events','LabResults','PharmacyData','UHSCOVIDTest','Vitalsigns_Cat', 'Vitalsigns_Num']

for study_id in demographics_df['STUDY_ID'].unique():
    data[study_id] = dict()
    
    # assign demographics information to each patient
    data[study_id] = assign_demographics(data[study_id], demographics_df, study_id)
    
    data[study_id] = compute_ICU_days(data[study_id], events_df, study_id)
    
    data[study_id] = assign_events(data[study_id], events_df, study_id)


# In[25]:


def check_if_died(dict_, covid_test_df, STUDY_ID, info = False):
    
    try:
        dict_['Died_in_2020'] = int(pd.notna(covid_test_df.loc[(covid_test_df['STUDY_ID'] == STUDY_ID),'YEAR_OF_DEATH']).reset_index(drop=True).values[0])
        if info:
            print('Died_in_2020? ', dict_['Died_in_2020'])
    except:
        if info:
            print('--------   error   -----------------')
    
    return dict_


# In[26]:


def get_lab_result(dict_, lab_results_df, STUDY_ID, info = False):
    
        
    # dict_ - a dictionary corresponding to a single patient
    # lab_results_df - a dataframe with lab results info directly from the csv file (datetime formating needed first)
    # STUDY_ID - patient id    
    
    ## Function adds to the dictionary a timeseries of lab results for given REACT_TESTCODE (e.g. POTASSIUM).
    ## Each REACT_TESTCOPE goes into dictionary as a new key. Values for each key are: time and values. 'time' is a timeline in DAYS, starting from 0. 'values' are the parameter values at given time.
    ## 'time' and 'values' are combined into a np.array() and assign to the key
    
    
    test_codes = list(lab_results_df['REACT_TESTCODE'].unique())
    
    for test_code in test_codes:
        #print(test_code)
        try:
            index = (lab_results_df['STUDY_ID'] == study_id) & (lab_results_df['REACT_TESTCODE'] == test_code)
            time = (lab_results_df.loc[index , 'PATHOLOGY_SPECIMEN_DATE'].reset_index(drop=True).values - lab_results_df.loc[index , 'PATHOLOGY_SPECIMEN_DATE'].reset_index(drop=True).values.min())/ np.timedelta64(1, 'D')
            order = np.argsort(time)
            time = time[order]
            values = lab_results_df.loc[index , 'PATHOLOGY_RESULT_NUMERIC'].reset_index(drop=True).values
            values = values[order]
            
            

            dict_[test_code] = np.array([time,values])
            if info:
                print('Patient ID ', study_id)
        except:
            if info:
                print('--------   error   -----------------')

    return dict_


# In[ ]:





# In[ ]:




for study_id in demographics_df['STUDY_ID'].unique():
    
    #print(study_id, '______________')
    
    data[study_id]['days_in_hospital'] , data[study_id]['discharged']= compute_days_in_hospital(data[study_id])
    
    data[study_id] = check_if_died(data[study_id], covid_test_df, study_id, False)
    
    data[study_id] = get_lab_result(data[study_id], lab_results_df, study_id, False)


# In[ ]:


data[study_id]


# In[ ]:


df = pd.DataFrame(data).T
df[(df['ITU'] == 1) & (df['discharged'] == 1)]


# In[ ]:


test_codes = list(lab_results_df['REACT_TESTCODE'].unique())
for test_code in test_codes:# = 'POTASSIUM'

    fig, ax = plt.subplots(figsize=(15,5))

    time_icu = np.array([])
    values_icu = np.array([])

    time_noicu = np.array([])
    values_noicu = np.array([])


    for patient in data.keys():

        try:
            #print(patient, '____\n', data[patient][test_code][0])

            time = data[patient][test_code][0]
            values = data[patient][test_code][1]

            icu_true = data[patient]['ITU']

            if icu_true==1:
                ax.plot(time, values,  '.r', linewidth =4, alpha = 1)
                time_icu = np.hstack((time_icu, time))
                values_icu = np.hstack((values_icu, values))
            else:
                ax.plot(time, values, '.b', linewidth =4, alpha = 1)
                time_noicu = np.hstack((time_noicu, time))
                values_noicu = np.hstack((values_noicu, values))

        except:
            pass
    ax.set_xlim([0, 30])
    ax.set_title(test_code);
    
    icu =  pd.DataFrame()
    icu['time'] = time_icu
    icu['values'] = values_icu
    icu = icu.sort_values(by = 'time').reset_index(drop=True)
    coefs = np.polyfit(icu['time'], icu['values'] , 4)
    p = np.poly1d(coefs)
    #plt.plot(icu['time'], icu['values'], "bo", markersize= 2)
    plt.plot(icu['time'], p(icu['time']), "r-")
    #plt.title(test_code + 'ICU')

    noicu =  pd.DataFrame()
    noicu['time'] = time_noicu
    noicu['values'] = values_noicu
    noicu = noicu.sort_values(by = 'time').reset_index(drop=True)
    coefs = np.polyfit(noicu['time'], noicu['values'] , 4)
    p = np.poly1d(coefs)
    #plt.plot(icu['time'], icu['values'], "bo", markersize= 2)
    plt.plot(noicu['time'], p(noicu['time']), "b-")


# In[ ]:


icu =  pd.DataFrame()
icu['time'] = time_icu
icu['values'] = values_icu
icu = icu.sort_values(by = 'time').reset_index(drop=True)


# In[ ]:


coefs = np.polyfit(icu['time'], icu['values'] , 4)
p = np.poly1d(coefs)
plt.plot(icu['time'], icu['values'], "bo", markersize= 2)
plt.plot(icu['time'], p(icu['time']), "r-")


# In[ ]:


sns.lineplot(x="time", y="values",
             #style="event",
             data=icu)


# In[ ]:


# df.to_csv(r'C:\Users\d07321ow\Google Drive\Cytokine\COVID19\REACT_data_processed\df_01.csv')
df.to_csv(rf'C:\work\dev\dECMT_src\Covid19_SH\src\df_01_oskar.csv')
print("==========  Finished ===========")

# In[ ]:





# In[ ]:




