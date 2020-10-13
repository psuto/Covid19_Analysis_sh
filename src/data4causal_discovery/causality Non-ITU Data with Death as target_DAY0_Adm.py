import numpy as np
import pandas as pd
import pathlib as p
import seaborn as sns
from pathlib import Path

def main():
    dataDirPath = "C:\Work\dev\dECMT_src\data_all\COVID19_Data\Current"
    fNnonITU = "labs_noITU.csv"
    fNnonITUday0 = "labs_noITU_day0.csv"
    fPNonITUday0 = Path(dataDirPath) / fNnonITUday0
    fPNonITU = Path(dataDirPath) / fNnonITU
    dfNonITUday0 = pd.read_csv(fPNonITUday0)
    dfNonITU = pd.read_csv(fPNonITU)
    print(list(dfNonITUday0.columns))
    # ['Unnamed: 0', 'STUDY_ID', 'PARAMETER', 'LOWER_RANGE', 'UPPER_RANGE', 'DATE', 'meanValue', 'category',
    # 'Days_since_ADMISSION', 'Days_since_COVID_FIRST_POSITIVE', 'Days_since_C5', 'Days_since_ITU',
    # 'Days_since_RESPIRATORY_HDU', 'Days_since_NIV', 'Days_since_INVASIVE_VENTILATION', 'Days_since_DISCHARGE',
    # 'Days_since_READMISSION', 'Days_since_DEATH', 'ADMISSION_START_DAY', 'ADMISSION_END_DAY', 'ITU_START_DAY',
    # 'ITU_END_DAY', 'INVASIVE_VENTILATION_START_DAY', 'INVASIVE_VENTILATION_END_DAY', 'NIV_START_DAY', 'NIV_END_DAY',
    # 'C5_START_DAY', 'C5_END_DAY', 'PATIENT_AGE', 'GENDER', 'DEATH_DAY', 'DISCHARGE_DAY', 'ITU', 'INV', 'NIV', 'C5',
    # 'DEATH', 'DISCHARGED']
    selectedColsDay0 = ['STUDY_ID',]
    print(len(dfNonITUday0))
    print(dfNonITUday0.value_counts('STUDY_ID'))

    pass

if __name__ == '__main__':
    main()