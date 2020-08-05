from datetime import datetime
import numpy as np
import pandas as pd
import pathlib as pl
import seaborn as sns



invAirCategoryDict=dict()
invAirCategoryDict[0]=set(['Air - Not Supported'])
invAirCategoryDict[1]=set(
    ['Nasal Specs', 'Face Mask', 'Supine',
       'Left', 'Venturi Mask', 'Non-Rebreath Mask',
       'Invasive Ventilation', 'Paralysed', 'Right', 'Unable to Assess',
       'Prone', 'Deep Sedation', 'Unrousable', 'Moderate Sedation',
       'Cardiac Chair Position', 'Optiflow / Hi Flow', 'Drowsy',
       'Sat in chair', 'Positive', 'Alert and Calm',
       'NIV - CPAP face mask', 'Restless', 'Asleep',
       'NIV - CPAP full face mask', 'Negative', 'Light Sedation',
       'NIV - BIPAP face mask',
       'Declined Repositioning - Risks explained to patient', 'Agitated',
       'Combative', 'Very Agitated', 'NIV - CPAP nasal mask',
       'NIV - BIPAP nasal mask', 'NIV - BIPAP full face mask',
       'CPAP - Green', 'Trachy Mask', 'Off ward for procedure',
       'Pressure Support - Blue', 'Trachy Mask - Red',
       'Speaking Valve - Yellow'
     ]
        )

airCategoryDict = {v:k for k,lnames in invAirCategoryDict.items() for v in lnames}


def extractParentDir(dataFileName):
    purePath = pl.PurePath(dataFileName)
    parentDir2 = purePath.parent.parent
    return parentDir2



fileName2VarDict = {
    "REACT_Comorbidities_unpivoted.csv": [ "renal_d"]
}



def getAggValForPatient(aggFnc, col4Value, data, data_dir, tableFileName, newAggregatedVarName, colValue):
    """
    Get aggregated value for feature in table
    :param aggFnc:
    :param col4Value:
    :param data:
    :param data_dir:
    :param tableFileName:
    :param newAggregatedVarName:
    :return:
    """
    comorbidDF = data.get(tableFileName)
    if comorbidDF is None:
        comorbidFN = comorbidFN = pl.Path(data_dir) / tableFileName
        comorbidDF: pd.DataFrame = pd.read_csv(comorbidFN)
        data[tableFileName] = comorbidDF
    comorbidDF[colValue] = (comorbidDF[col4Value] == colValue)
    # Index(['STUDY_ID', 'COMORBIDITY', 'STATUS'], dtype='object')
    # renalDDF = comorbidDF.groupby('STUDY_ID').agg(RENAL_D_PRESENT=('RENAL_D_PRESENT', 'any'))
    renalDDF = comorbidDF.groupby('STUDY_ID').agg(
        **{newAggregatedVarName: pd.NamedAgg(column=col4Value, aggfunc=aggFnc)})
    return renalDDF


def getRenalDData(data_dir,data:dict):
    fn  = 'REACT_Comorbidities_unpivoted.csv'
    col4Value = 'COMORBIDITY'
    colValue = 'RENAL_D_PRESENT'
    newAggregatedVarName = 'RENAL_D_PRESENT'
    aggFnc = 'any'
    renalDDF = getAggValForPatient(aggFnc, col4Value, data, data_dir, fn, newAggregatedVarName,colValue)
    return renalDDF


def getMalignantNeo(data_dir,data:dict):
    fn = 'REACT_Comorbidities_unpivoted.csv'
    col4Value = 'COMORBIDITY'
    colValue = 'RENAL_D_PRESENT'
    newAggregatedVarName = 'RENAL_D_PRESENT'
    aggFnc = 'any'
    renalDDF = getAggValForPatient(aggFnc, col4Value, data, data_dir, fn, newAggregatedVarName, colValue)
    return renalDDF


def getDeath(data_dir):
    fn = "REACT_Demographics.csv"
    dfFileName = pl.Path(data_dir) / fn
    df: pd.DataFrame = pd.read_csv(dfFileName)
    df['Death'] = df['DATE_OF_DEATH'].notna()
    dfRes = df.groupby('STUDY_ID').agg(Death=('Death', 'any'))
    return dfRes


def getRespiratorySupportDF(data_dir):
    # "REACT_Vitalsigns_Categorical.csv"
    fn = "REACT_Vitalsigns_Categorical.csv"
    dfFileName = pl.Path(data_dir) / fn
    df: pd.DataFrame = pd.read_csv(dfFileName)
    df = df[df.PARAMETER == 'Respiratory Support'][['STUDY_ID', 'UNITFROM_DATETIME', 'VALUE']]
    df["AIR_SUPPORT"] = df["VALUE"].map(airCategoryDict)
    df["AIR_SUPPORT"] = df["AIR_SUPPORT"] == 1
    df = df.filter(items=['STUDY_ID', 'AIR_SUPPORT'])
    dfRes = df.groupby('STUDY_ID').agg(AIR_SUPPORT=('AIR_SUPPORT', 'any'))
    return dfRes

todayVal = datetime.today()
timeStampStr = todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")

if __name__ == '__main__':


    def main():
        data_dir = r"C:\work\dev\dECMT_src\data_all\COVID19_Data\Current"
        parentDir2 = extractParentDir(data_dir)
        timeStampStr = todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")
        data = dict()

        outputPath = pl.Path(parentDir2) / "Output_Covid19_Causal_Data" # / f"{timeStampStr}_causal_data4Tetrad"
        pl.Path(outputPath).mkdir(exist_ok=True)
        renalDDF = getRenalDData(data_dir,data)
        malignantNeoDF = getMalignantNeo(data_dir,data)
        deathDF = getDeath(data_dir)
        respSupportDF = getRespiratorySupportDF(data_dir)
        joinedDF = pd.merge(renalDDF,deathDF,on='STUDY_ID', how='inner')
        resDF = pd.merge(joinedDF,respSupportDF,on='STUDY_ID', how='inner')
        outputFile = pl.Path(outputPath) /f"{timeStampStr}_TetradData_renalD_death_respSuppBinary.csv"
        resDF.to_csv(outputFile,index=False)
        print(f"The Output written to {outputFile}")
        pass


    main()
