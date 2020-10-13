import pandas as pd
from pathlib import Path
import csv
import numpy as np
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import pickle
import json
from sklearn.model_selection import train_test_split
import math
import utilities

air_ventilator_degree=[
    'Air - Not Supported',
    'Nasal Specs',
    'Face Mask',
    'Venturi Mask',
    'Trachy Mask',
    'Non-Rebreath Mask',
    'Optiflow / Hi Flow',
    'NIV - CPAP nasal mask',
    'NIV - CPAP face mask',
    'NIV - CPAP full face mask',
    'NIV - BIPAP nasal mask',
    'NIV - BIPAP face mask',
    'NIV - BIPAP full face mask',
    'Invasive Ventilation'
]


def readFiles(path):
    demographics_df = pd.read_csv(path + '\REACT_COVID_Demographics_20200506.csv')
    events_df = pd.read_csv(path + '\REACT_Events.csv')
    lab_results_df = pd.read_csv(path + '\REACT_LabResults.csv')
    pharmacy_data_df = pd.read_csv(path + '\REACT_PharmacyData.csv')
    covid_test_df = pd.read_csv(path + '\REACT_UHSCOVIDTest_processed.csv')
    vitalsigns_cat_df = pd.read_csv(path + '\REACT_Vitalsigns_Categorical.csv')
    vitalsigns_num_df = pd.read_csv(path + '\REACT_Vitalsigns_Numeric.csv')
    res = {'demographics_df': demographics_df, 'events_df': events_df, 'lab_results_df': lab_results_df,
           'pharmacy_data_df': pharmacy_data_df, 'covid_test_df':covid_test_df,
           'vitalsigns_cat_df':vitalsigns_cat_df,'vitalsigns_num_df':vitalsigns_num_df}
    return res


def categoryToNumericAsDict(categories):
    res = dict([(air_ventilator_degree[i], float(i)) for i in range(len(air_ventilator_degree))])
    return res

air_ventilator_degree_dic=categoryToNumericAsDict(air_ventilator_degree)

def main():
    dirPath = r'C:\work\dev\dECMT_src\data_all\COVID19_Data'
    dfs = readFiles(dirPath)
    print(air_ventilator_degree_dic)
    # === Test this code ===========================

    studyid_air_ventilator_dic = dict()

    # for studyid, air_vent in     df_REACT_Vitalsigns_Categorical[df_REACT_Vitalsigns_Categorical.VALUE.isin(air_ventilator_degree_dic)][
    #     ['STUDY_ID', 'VALUE']].values:
    #     if studyid not in studyid_air_ventilator_dic or studyid_air_ventilator_dic[studyid] < air_ventilator_degree_dic[
    #         air_vent]:
    #         studyid_air_ventilator_dic[studyid] = air_ventilator_degree_dic[air_vent]

    df_studyid_air_vent = pd.DataFrame(
        data=[[key, studyid_air_ventilator_dic[key]] for key in studyid_air_ventilator_dic],
        columns=['STUDY_ID', 'AIR_VENT_DEGREE']
    )
    # ================================

    print()

if __name__ == '__main__':
    main()