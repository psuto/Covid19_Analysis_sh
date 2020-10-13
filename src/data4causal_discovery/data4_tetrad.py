from datetime import datetime
import numpy as np
import pandas as pd
import pathlib as pl
import seaborn as sns
from pathlib import Path
import re
import covid

todayVal = datetime.today()
timeStampStr = todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")
timeStampStr = todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")

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
    origDF = data.get(tableFileName)
    if origDF is None:
        comorbidFN = comorbidFN = pl.Path(data_dir) / tableFileName
        origDF: pd.DataFrame = pd.read_csv(comorbidFN)
        data[tableFileName] = origDF
    origDF[colValue] = (origDF[col4Value] == colValue)
    # Index(['STUDY_ID', 'COMORBIDITY', 'STATUS'], dtype='object')
    # renalDDF = comorbidDF.groupby('STUDY_ID').agg(RENAL_D_PRESENT=('RENAL_D_PRESENT', 'any'))
    resultDF = origDF.groupby('STUDY_ID').agg(
        **{newAggregatedVarName: pd.NamedAgg(column=col4Value, aggfunc=aggFnc)})
    return resultDF

def getAggEventWDateForPatient(aggFnc, col4Value, data, data_dir, tableFileName, newAggregatedVarName, colValue):
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
    origDF = data.get(tableFileName)
    if origDF is None:
        comorbidFN = comorbidFN = pl.Path(data_dir) / tableFileName
        origDF: pd.DataFrame = pd.read_csv(comorbidFN)
        data[tableFileName] = origDF
    origDF[colValue] = (origDF[col4Value] == colValue)
    # Index(['STUDY_ID', 'COMORBIDITY', 'STATUS'], dtype='object')
    # renalDDF = comorbidDF.groupby('STUDY_ID').agg(RENAL_D_PRESENT=('RENAL_D_PRESENT', 'any'))
    resultDF = origDF.groupby('STUDY_ID').agg(
        **{newAggregatedVarName: pd.NamedAgg(column=col4Value, aggfunc=aggFnc)})
    return resultDF


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
    df1 = df.filter(items=['STUDY_ID','DATE_OF_DEATH','Death'])
    assert len(df1)==len(df1['STUDY_ID'].unique()), "Not unique STUDY_ID problem !!!"
    dfRes = df1
    # print(len(df1))
    # print(len(df1['STUDY_ID'].unique()))
    # dfRes = df.groupby('STUDY_ID').agg(Death=('Death', 'any'), DATE_OF_DEATH=('DATE_OF_DEATH','first'))
    # dfRes.reset_index(inplace=True)
    return dfRes

def getDischargeDate(data_dir):
    fn = "REACT_Demographics.csv"
    dfFileName = pl.Path(data_dir) / fn
    df: pd.DataFrame = pd.read_csv(dfFileName)
    df['Discharged'] = df['DISCHARGEDATE'].notna()
    df1 = df.filter(items=['STUDY_ID','DISCHARGEDATE','Discharged'])
    dfRes =df1
    # dfRes = df.groupby('STUDY_ID').agg(Death=('Death', 'any'), DATE_OF_DEATH=('DATE_OF_DEATH','first'))
    # dfRes.reset_index(inplace=True)
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

def getITUFlag(data_dir, data):
    fn = 'REACT_Events.csv'
    col4Value = 'EVENT_TYPE'
    colValue = 'ITU'
    newAggregatedVarName = 'ITU'
    aggFnc = 'any'
    ituDF = getAggValForPatient(aggFnc, col4Value, data, data_dir, fn, newAggregatedVarName, colValue)
    return ituDF

def getNonITUFStudyIDs(data_dir, data, tableFileName='REACT_Events'):
    origDF = data.get(tableFileName)
    if origDF is None:
        comorbidFN = comorbidFN = pl.Path(data_dir) / f"{tableFileName}.csv"
        origDF: pd.DataFrame = pd.read_csv(comorbidFN)
        data[tableFileName] = origDF
    # origDF[colValue] = (origDF[col4Value] == colValue)
    origDF['ITU_Admission'] = origDF.EVENT_TYPE == 'ITU'
    dfRes:pd.DataFrame = origDF.loc[~origDF['ITU_Admission'], 'STUDY_ID']
    # dfRes1 = dfRes.groupby('STUDY_ID').first()
    len(dfRes.value_counts())==len(dfRes)
    len(dfRes.unique())==len(dfRes)
    vc = dfRes.value_counts()
    un =dfRes.unique()
    dfRes1 = pd.DataFrame(data = dfRes.unique(),columns=['STUDY_ID'])
    len(dfRes1['STUDY_ID'].unique()) == len(dfRes1)
    return dfRes1


def getITUFlagWithDates(data_dir, data, tableFileName='REACT_Events'):
    # df_event=pd.read_csv(data_dir+'REACT_Events'+'.csv')
    origDF = data.get(tableFileName)
    if origDF is None:
        comorbidFN = comorbidFN = pl.Path(data_dir) / f"{tableFileName}.csv"
        origDF: pd.DataFrame = pd.read_csv(comorbidFN)
        data[tableFileName] = origDF
    # origDF[colValue] = (origDF[col4Value] == colValue)
    origDF['ITU_Admission'] = origDF.EVENT_TYPE == 'ITU'
    origDF['START_DATETIME_ITU'] = pd.NA
    origDF['END_DATETIME_ITU'] = pd.NA
    origDF.loc[origDF['ITU_Admission'],'START_DATETIME_ITU'] = origDF['START_DATETIME']
    origDF.loc[origDF['ITU_Admission'],'END_DATETIME_ITU'] = origDF['END_DATETIME']
    dfRes = origDF
    # dfRes:pd.DataFrame = origDF.filter(items=['STUDY_ID', 'ITU_Admission', 'START_DATETIME','END_DATETIME'])
    # dfRes.rename(columns={'START_DATETIME':'START_DATETIME_ITU','END_DATETIME':'END_DATETIME_ITU'},inplace=True)
    return dfRes


def getITUDatesDischarges(data_dir, data):
    # getITUDatesDischarges
    # REACT_Events.csv
    fn = 'REACT_Events.csv'
    col4Value = 'EVENT_TYPE'
    colValue = 'ITU'
    newAggregatedVarName = 'ITU'
    aggFnc = 'any'
    ituDF = getAggValForPatient(aggFnc, col4Value, data, data_dir, fn, newAggregatedVarName, colValue)
    return ituDF


    # fn = "REACT_Demographics.csv"
    # dfFileName = pl.Path(data_dir) / fn
    # df: pd.DataFrame = pd.read_csv(dfFileName)
    # df['Death'] = df['DATE_OF_DEATH'].notna()
    # dfRes = df.groupby('STUDY_ID').agg(Death=('Death', 'any'))
    # return dfRes


class DataTransformerGeneral:
    def __init__(self):
        pass

class DataTransformerLabsSince_DoD_or_Adm(DataTransformerGeneral):
    def __init__(self, dir):
        super(self).__init__()
        self._dir = dir

    @property
    def dir(self):
        return self._dir

    def transform(self):

        pass


class DataTransformerLabsSince_DoD_or_Adm_00(DataTransformerGeneral):
    """
    Original without preprocessed lab_noITU
    """
    def __init__(self, dataDir):
        super(self).__init__()
        self._dataDir = dataDir

    def printInfo(self):
        print('Not using lab_noITU - using baseline information')

    @property
    def dataDir(self):
        return self._dataDir

    def transform(self):
        data_dir=self._dataDir
        parentDir2 = extractParentDir(data_dir)
        outputPath = Path(parentDir2) / "Output_Covid19_Causal_Data"  # / f"{timeStampStr}_causal_data4Tetrad"
        Path(outputPath).mkdir(exist_ok=True)
        timeStampStr =todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")
        data = dict()
        # Non ITU
        dfNonITU = getNonITUFStudyIDs(data_dir, data)
        # Death
        deathDF = getDeath(data_dir)
        deathDF.reset_index(inplace=True)
        # Discharge Date
        df_Death_ITU = pd.merge(deathDF, dfNonITU, on='STUDY_ID', how='inner')
        # Labs
        dataDirPath = "C:\Work\dev\dECMT_src\data_all\COVID19_Data\Current"
        fnLabs = 'REACT_LabResults.csv'
        fPNonITUday0DoD = Path(dataDirPath) / fnLabs
        dfNonITUday0DoD = pd.read_csv(fPNonITUday0DoD)




class DataTransformer01LabsSince_DoD_or_Adm_01(DataTransformerGeneral):
    """
    Using lab_noITU
    """
    def printInfo(self):
        print('Using lab_noITU')

    def __init__(self, dataDir):
        super().__init__()
        self._dataDir = dataDir
        parentDir2 = extractParentDir(dataDir)
        outputPath = pl.Path(parentDir2) / "Output_Covid19_Causal_Data"  # / f"{timeStampStr}_causal_data4Tetrad"
        pl.Path(outputPath).mkdir(exist_ok=True)
        print(f"Output path {outputPath}")
        self.outputDir = outputPath

    @property
    def dataDir(self):
        return self._dataDir

    def paramNameWithDay(self, dfNonITU):
        # dfNonITU['Parameter_and_Day'] = dfNonITU['PARAMETER'] + dfNonITU['Days_FromEvent']
        cs = []
        for _, row in dfNonITU.iterrows():
            x = f"{row['PARAMETER']}_{row['Days_FromEvent']:n}"
            cs.append(x)
            pass
        dfNonITU['Parameter_and_Day'] = cs
        return dfNonITU

    def fixColName(self, s):
            res = ""
            # 'Categ_mean_ALT_1'
            m = re.search("(^Categ_mean_)(.*)",s)
            res = f"{m.group(2)}(c_mean)"
            return res

    def adjustdfNonITU(self,dfNonITU):
        # ['STUDY_ID', 'PARAMETER', 'LOWER_RANGE', 'UPPER_RANGE', 'DATE', 'meanValue', 'category', 'Days_since_DEATH',
        #  'Days_since_DISCHARGE', 'Event_Type_Death']
        dfNonITU['Event_Type_Death'] = -1
        dfNonITU['Days_FromEvent'] = np.nan
        dfNonITU['Parameter_and_Day'] = np.nan
        dfNonITU['48hxN'] = np.nan
        categMapping = {'NORMAL':1, 'LOW':0, 'HIGH':2 }
        invCategMapping = {v:k for k,v in categMapping.items()}
        dfNonITU['categNum'] = dfNonITU['category'].map(categMapping)

        # ['NORMAL' 'LOW' 'HIGH' nan]
        dfNonITU.loc[~dfNonITU['Days_since_DISCHARGE'].isna(),'Event_Type_Death'] = 0
        x0 = dfNonITU['Event_Type_Death'].value_counts()
        dfNonITU.loc[~dfNonITU['Days_since_DEATH'].isna(), 'Event_Type_Death'] = 1
        x1 = dfNonITU['Event_Type_Death'].value_counts()
        # ******************************************************************************************************
        dfNonITU.loc[dfNonITU['Event_Type_Death']==1,'Days_FromEvent'] = abs(dfNonITU.loc[dfNonITU['Event_Type_Death']==1,'Days_since_DEATH'])
        dfNonITU.loc[dfNonITU['Event_Type_Death']==0,'Days_FromEvent'] = abs(dfNonITU.loc[dfNonITU['Event_Type_Death']==0,'Days_since_DISCHARGE'])
        # ******************************************************************************************************
        dfNonITU = self.paramNameWithDay(dfNonITU)
        # ******************************************************************************************************
        x2 = len(dfNonITU['STUDY_ID'].unique())
        # '48hxN'
        map48h = (dfNonITU['Days_FromEvent']>=0) & (dfNonITU['Days_FromEvent']<2)
        dfNonITU.loc[map48h,'48hxN']= 1
        dfNonITUSel = dfNonITU[map48h]
        # ['STUDY_ID', 'PARAMETER', 'LOWER_RANGE', 'UPPER_RANGE', 'DATE', 'meanValue', 'category', 'Days_since_DEATH',
        # 'Days_since_DISCHARGE', 'Event_Type_Death', 'Days_FromEvent', 'Parameter_and_Day', '48hxN']
        def reconcile(x):
            print(x)

        grp1:pd.DataFrame = dfNonITUSel.groupby(['STUDY_ID','Parameter_and_Day','Event_Type_Death']).agg(MeanValue_mean=('meanValue','mean'),
                                                                                      MeanValue_first=('meanValue','first'),
                                                                                      MeanValue_max=('meanValue','max'),
                                                                                      MeanValue_count=('meanValue','count'),
                                                                                      Category_first=('category', 'first'),
                                                                                      Category_last=('category', 'last'),
                                                                                      Category_mean=('categNum','mean'),
                                                                                      Category_max = ('categNum', 'max')
                                                                                      )
        mp = grp1['MeanValue_mean']!=grp1['MeanValue_first']
        xxx = grp1[mp]

        grp2 = grp1.reset_index()
        # ['STUDY_ID', 'PARAMETER', 'LOWER_RANGE', 'UPPER_RANGE', 'DATE', 'meanValue', 'category', 'Days_since_DEATH',
        # 'Days_since_DISCHARGE', 'Event_Type_Death', 'Days_FromEvent', 'Parameter_and_Day', '48hxN']
        dfNonITUF0Categ = pd.pivot_table(grp2,index = ['STUDY_ID','Event_Type_Death'],values=['Category_mean','Category_max'],columns='Parameter_and_Day',aggfunc=['max','mean'])
        # dfNonITUF0Categ = pd.pivot(grp2,index = ['STUDY_ID','Event_Type_Death'],values='Category_first',columns='Parameter_and_Day')
        # dfNonITUF0Conti = pd.pivot(grp2,index = ['STUDY_ID','Event_Type_Death'],values='MeanValue_mean',columns='Parameter_and_Day')
        dfNonITUF0Conti = pd.pivot_table(grp2,index = ['STUDY_ID','Event_Type_Death'],values='MeanValue_mean',columns='Parameter_and_Day',aggfunc=['max','mean'])
        dfNonITUF0CategGrp = grp2.groupby(['STUDY_ID','Event_Type_Death','Parameter_and_Day']).agg(Categ_max=('Category_max','max'),Categ_mean=('Category_mean','mean'))
        dfNonITUF0CategGrp.reset_index(inplace=True)
        dfNonITUF0CategGrp['Categ_mean'] = round(dfNonITUF0CategGrp['Categ_mean'])
        dfNonITUF0CategGrp['Categ_max'] = dfNonITUF0CategGrp['Categ_max'].map(invCategMapping)
        dfNonITUF0CategGrp['Categ_mean'] = dfNonITUF0CategGrp['Categ_mean'].map(invCategMapping)
        dfNonITUF0Categ2 = pd.pivot_table(dfNonITUF0CategGrp,index = ['STUDY_ID','Event_Type_Death'],values=['Categ_max','Categ_mean'],columns='Parameter_and_Day',aggfunc='first')
        dfNonITUF0Categ2.reset_index(inplace=True)
        cols1 = ['_'.join(col).strip() for col in dfNonITUF0Categ2.columns.values]
        dfNonITUF0Categ2.columns = ['_'.join(col).strip() for col in dfNonITUF0Categ2.columns.values]
        dfNonITUF0Categ2.rename(columns={'STUDY_ID_':'STUDY_ID', 'Event_Type_Death_':'Event_Type_Death'}, inplace=True)
        # ['STUDY_ID_', 'Event_Type_Death_', 'Categ_max_ALT_1', 'Categ_max_AST_1', 'Categ_max_BILIRUBIN_1', 'Categ_max_CREATENINE_1', 'Categ_max_CRP_1', 'Categ_max_D_DIMER_1', 'Categ_max_EOSINOPHILS_1', 'Categ_max_FERRITIN_1', 'Categ_max_GLUCOSE_1', 'Categ_max_HB_1', 'Categ_max_LDH_1', 'Categ_max_LYMPHOCYTES_1', 'Categ_max_NEUTROPHILS_1', 'Categ_max_PLATELETS_1', 'Categ_max_POTASSIUM_1', 'Categ_max_SODIUM_1', 'Categ_max_UREA_1', 'Categ_max_WBC_1', 'Categ_mean_ALT_1', 'Categ_mean_AST_1', 'Categ_mean_BILIRUBIN_1', 'Categ_mean_CREATENINE_1', 'Categ_mean_CRP_1', 'Categ_mean_D_DIMER_1', 'Categ_mean_EOSINOPHILS_1', 'Categ_mean_FERRITIN_1', 'Categ_mean_GLUCOSE_1', 'Categ_mean_HB_1', 'Categ_mean_LDH_1', 'Categ_mean_LYMPHOCYTES_1', 'Categ_mean_NEUTROPHILS_1', 'Categ_mean_PLATELETS_1', 'Categ_mean_POTASSIUM_1', 'Categ_mean_SODIUM_1', 'Categ_mean_UREA_1', 'Categ_mean_WBC_1']
        cols1Sel = [x for x in cols1 if x.startswith('Categ_mean_')]

        cols1SelNew = {x: self.fixColName(x) for x in cols1Sel}
        dfNonITUF0Categ2.rename(columns = cols1SelNew,inplace=True)
        selCols = list(dfNonITUF0Categ2.columns)[:2]+ [v for k,v in cols1SelNew.items()]
        dfNonITUF0Categ2 = dfNonITUF0Categ2.filter(items = selCols)
        dfNonITUF0Categ2.fillna(value="Missing")
        # df2 = dfNonITUF0['meanValue'].reset_index()
        dfNonITUF0Categ = dfNonITUF0Categ.reset_index()
        dfNonITUF0Conti = dfNonITUF0Conti.reset_index()
        # dfNonITUF0Categ = dfNonITUF0Categ.columns.name  = None

        # dfNonITUDay0 =
        # y = dfNonITU['Event_Type_Death'].value_counts()
        return {'categorical':dfNonITUF0Categ2,'continuous':dfNonITUF0Conti}

    def transform(self):
        data_dir=self._dataDir
        parentDir2 = extractParentDir(data_dir)
        outputPath = Path(parentDir2) / "Output_Covid19_Causal_Data"  # / f"{timeStampStr}_causal_data4Tetrad"
        print(f"Input path : {data_dir}")
        print(f"Output path : {outputPath}")
        Path(outputPath).mkdir(exist_ok=True)
        timeStampStr =todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")
        data = dict()
        # Non ITU
        dfNonITU = getNonITUFStudyIDs(data_dir, data)
        # Death
        deathDF = getDeath(data_dir)
        deathDF.reset_index(inplace=True)
        # Discharge Date
        df_Death_ITU = pd.merge(deathDF, dfNonITU, on='STUDY_ID', how='inner')
        # Labs
        dataDirPath = "C:\Work\dev\dECMT_src\data_all\COVID19_Data\Current"
        labsPreprocessed = 'labs_noITU.csv'
        fPNonITU = Path(dataDirPath) / labsPreprocessed
        dfNonITU0 = pd.read_csv(fPNonITU)
        cids = len(dfNonITU0['STUDY_ID'].unique())
        # Selected files
        # ['Unnamed: 0', 'STUDY_ID', 'PARAMETER', 'LOWER_RANGE', 'UPPER_RANGE', 'DATE', 'meanValue', 'category',
        # 'Days_since_ADMISSION', 'Days_since_COVID_FIRST_POSITIVE', 'Days_since_C5', 'Days_since_ITU',
        # 'Days_since_RESPIRATORY_HDU', 'Days_since_NIV', 'Days_since_INVASIVE_VENTILATION', 'Days_since_DISCHARGE',
        # 'Days_since_READMISSION', 'Days_since_DEATH', 'ADMISSION_START_DAY', 'ADMISSION_END_DAY', 'ITU_START_DAY',
        # 'ITU_END_DAY', 'INVASIVE_VENTILATION_START_DAY', 'INVASIVE_VENTILATION_END_DAY', 'NIV_START_DAY', '
        # NIV_END_DAY', 'C5_START_DAY', 'C5_END_DAY', 'PATIENT_AGE', 'GENDER', 'DEATH_DAY', 'DISCHARGE_DAY', 'ITU',
        # 'INV', 'NIV', 'C5', 'DEATH', 'DISCHARGED']
        self.dfNonITU0 = dfNonITU0.filter(items=[ 'STUDY_ID', 'PATIENT_AGE', 'GENDER'])
        self.dfNonITU0['STUDY_ID'].value_counts()
        len(self.dfNonITU0['STUDY_ID'].unique())
        dfNonITU = dfNonITU0.filter(items=['STUDY_ID', 'PARAMETER', 'LOWER_RANGE', 'UPPER_RANGE', 'DATE', 'meanValue',
                                           'category','Days_since_DEATH','Days_since_DISCHARGE'])
        x = dfNonITU['Days_since_DEATH'].value_counts()
        y = dfNonITU['Days_since_DISCHARGE'].value_counts()
        res = self.adjustdfNonITU(dfNonITU)
        res = self.addDemog(res)
        self.res = res
        self.saveResults()
        return(res)

    def saveResults(self):
        res = self.res
        pass
        dfCateg = res.get('categorical')
        dfCont = res.get('continuous')
        p1 = Path(self.outputDir)
        fnCatag = p1 / f"nonITU_days_from_death_or_discharge_Categ_{timeStampStr}.csv"
        fnCont = p1 / f"nonITU_days_from_death_or_discharge_Cont_{timeStampStr}.csv"
        dfCont.to_csv(fnCont,index=False)
        dfCateg.to_csv(fnCatag,index=False)
        print(f"File saved to: {fnCatag}")
        print(f"File saved to: {fnCont}")

    def addDemog(self, res):
        path = pl.Path(self._dataDir)
        demog_data_df2 = pd.read_csv(path / 'REACT_COVID_Demographics_20200506.csv')
        demog_data_df = pd.read_csv(path / 'REACT_COVID_Demographics.csv')
        demog_data_df['STUDY_ID'].value_counts()
        # ['STUDY_ID', 'PATIENT_AGE', 'GENDER', 'ETHNIC_GROUP', 'POSTCODE', 'IS_PREGNANT', 'FIRST_POS_DATE',
        #  'FIRST_POS_DATETIME', 'FIRST_POS_TIME', 'ADM_DATE', 'ADM_DATETIME', 'ADM_TIME', 'DISCHARGE_DATE', 'LOS',
        #  'LOS_PREPOS']
        # selCols = ['STUDY_ID', 'PATIENT_AGE', 'GENDER', 'ETHNIC_GROUP', 'POSTCODE', 'IS_PREGNANT','LOS','LOS_PREPOS']
        selCols = ['STUDY_ID', 'PATIENT_AGE', 'GENDER', 'ETHNIC_GROUP', 'POSTCODE', 'IS_PREGNANT']
        demog_data_df  = demog_data_df.filter(items = selCols)
        demog_data_dfGrp = demog_data_df.groupby(by='STUDY_ID').first()
        demog_data_df2Grp = demog_data_df2.groupby(by='STUDY_ID').first()
        dfCateg = res.get('categorical')
        myres = pd.merge(dfCateg,demog_data_dfGrp,on='STUDY_ID',how='left')
        myres = pd.merge(myres,demog_data_df2Grp,on='STUDY_ID',how='left')
        myres.fillna(value='Missing',inplace=True)
        res['categorical']=myres
        myres['STUDY_ID'].value_counts()
        idCount = len(myres['STUDY_ID'].unique())
        shp = myres.shape
        return res


if __name__ == '__main__':

    def main():
        # data_dir = r"C:\work\dev\dECMT_src\data_all\COVID19_Data\Current"
        data_dir = r"C:\Work\DropBoxPS\Dropbox\dev\data\dECMT\COVID19_SH\Current"
        parentDir2 = extractParentDir(data_dir)

        outputPath = pl.Path(parentDir2) / "Output_Covid19_Causal_Data" # / f"{timeStampStr}_causal_data4Tetrad"
        pl.Path(outputPath).mkdir(exist_ok=True)
        data = dict()
        dfNonITU = getNonITUFStudyIDs(data_dir,data)
        ituFlag = getITUFlagWithDates(data_dir,data)
        deathDF = getDeath(data_dir)
        dischargeDate = getDischargeDate(data_dir)
        renalDDF = getRenalDData(data_dir,data)
        # todo: Finish this malignantNeoDF = getMalignantNeo(data_dir,data)
        # malignantNeoDF = getMalignantNeo(data_dir,data)
        # getScrTSAggVals(data_dir,data)
        respSupportDF = getRespiratorySupportDF(data_dir)

        joinedDF = pd.merge(renalDDF,deathDF,on='STUDY_ID', how='inner')
        resDF = pd.merge(joinedDF,respSupportDF,on='STUDY_ID', how='inner')
        resDF = pd.merge(resDF,ituFlag, on='STUDY_ID', how='inner')
        outputFile = pl.Path(outputPath) /f"{timeStampStr}_TetradData_renalD_death_respSuppBinary.csv"
        resDF.to_csv(outputFile,index=False)
        print(f"The Output written to {outputFile}")
        # getITUDatesDischarges(data_dir,data)



    main()
