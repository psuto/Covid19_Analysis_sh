import numpy as np
import pandas as pd
import pathlib as p
import seaborn as sns
from pathlib import Path
import data4_tetrad as dt

def main():
    data_dir = r"C:\Work\DropBoxPS\Dropbox\dev\data\dECMT\COVID19_SH\Current"
    # dtranf00 = dt.DataTransformerLabsSince_DoD_or_Adm_00(data_dir)
    # dtranf00.transform()

    dtranf01 = dt.DataTransformer01LabsSince_DoD_or_Adm_01(data_dir)
    dtranf01.printInfo()
    dtranf01.transform()




    print(f'')

if __name__ == '__main__':
    main()
