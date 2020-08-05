import pandas as pd
import utilities as u
import gps.preproces_Covid_gps as ppC
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def timeSeriesGradient(df,timeColName,sourceColName,targetColName):
    df[targetColName] = df.diff() / df[sourceColName].to_series().diff().dt.total_seconds()


def deriveFeatures(df):
    # derive gradients
    pass


def main():
    dirPath = "C:\work\dev\dECMT_src\data_all\COVID19_Data\Current"
    fp = r"C:\work\dev\dECMT_src\data_all\COVID19_Data\Current\po2_fo2_data20-06-22_10-09-05.869361.csv"
    df = pd.read_csv(fp)
    print(df.describe())
    list(df.columns)
    print(df.head())
    ids = df['STUDY_ID'].unique()
    print(ids)
    # ============================================
    df['ICU_Present'] = df['ICU_Days'] > 0
    df['Ventilation_Present'] = (df['NIV'] & df['INVASIVE VENTILATION'])
    # ============================================
    df = deriveFeatures(df)
    # ============================================
    mapICUD0 = df['ICU_Days'] <= 0
    mapICUDgt0 = df['ICU_Days'] > 0

    dfICUD0 = df[mapICUD0]
    dfICUDgt0 = df[mapICUDgt0]
    #=================================


    #=================================



    #=================================

    with sns.axes_style(style=None):
        sns.violinplot("ICU_Present", "pO2_FiO2", data=df,
                       split=True, inner="quartile",
                       palette=["lightblue", "lightpink"]);
        axes = plt.gca()
        axes.set_ylim([-100, 400])



if __name__ == '__main__':
    main()