import pandas as pd

def readData(dataFileN, nRows2REad):
    df1 = pd.DataFrame()
    if nRows2REad <= 0:
        df1 = pd.read_csv(dataFileN)
    else:
        df1 = pd.read_csv(dataFileN, nrows=nRows2REad)
    return df1