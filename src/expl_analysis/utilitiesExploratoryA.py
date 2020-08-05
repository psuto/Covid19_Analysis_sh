import pathlib
from datetime import datetime
import copy
import pandas as pd
import numpy as np


todayVal = datetime.today()
timeStampStr = todayVal.strftime("%y-%m-%d_%H-%M-%S.%f")


def readData(dataFileN, nRows2REad):
    df1 = pd.DataFrame()
    if nRows2REad <= 0:
        df1 = pd.read_csv(dataFileN)
    else:
        df1 = pd.read_csv(dataFileN, nrows=nRows2REad)
    return df1

def list_diff(list1, list2):
    res = list()
    if (len(list1)-len(list2))==0:
        res = list()
    elif len(list1)>0 and len(list2)>0:
        try:
            res = (list(set(list1) - set(list2)))
        except Exception as e:
            print(e)
    elif len(list1)>0:
        res = copy.deepcopy(list1)
    return res

def extractParentDir(dataFileName):
    purePath = pathlib.PurePath(dataFileName)
    parentDir2 = purePath.parent.parent
    return parentDir2