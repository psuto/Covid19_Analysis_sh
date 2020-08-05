import utilities as u
import pandas as pd

def main():
    fileName = f"C:\work\dev\dECMT_src\data_all\COVID19_Data\df_01_oskar.csv"
    df = u.readData(fileName)
    colNames = list(df.columns.values)
    # ['ID', 'GENDER', 'ETHNIC_GROUP', 'IS_PREGNANT', 'PATIENT_AGE', 'ADM_DATETIME', 'DISCHARGE_DATE',
    # 'events_hosp_days', 'C5', 'INVASIVE VENTILATION', 'ITU', 'NIV', 'days_in_hospital', 'discharged',
    # 'Died_in_2020', 'POTASSIUM', 'UREA', 'CREATENINE', 'SODIUM', 'NEUTROPHILS', 'EOSINOPHILS', 'HB', 'LYMPHOCYTES',
    # 'WBC', 'PLATELETS', 'GLUCOSE', 'AST', 'TROPONIN', 'D_DIMER', 'ALT', 'BILIRUBIN', 'CRP', 'LDH', 'TRIGYCERIN',
    # 'FERRITIN', 'ICU_days']
    print()
    #  ===========================================================================
    selectedCols = ['ID','GENDER', 'ETHNIC_GROUP', 'IS_PREGNANT', 'PATIENT_AGE']
    dfOut = pd.DataFrame(columns=[])
    # ToDo: ITU days df[1].fillna(0, inplace=True)

    print()

if __name__ == '__main__':
    main()