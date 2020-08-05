from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial
import pandas as pd
from pathlib import Path

if __name__ == "__main__":

    def main():
        lb = 21
        ub = 30
        outPathDir = r"C:\work\dev\dECMT_src\Outputs\Covid19_SH_Outputs\Plots"
        fp = r"C:\work\dev\dECMT_src\data_all\COVID19_Data\Current\po2_fo2_data20-06-22_10-09-05.869361.csv"
        df = pd.read_csv(fp)
        print(df.describe())
        colN = list(df.columns)
        print(colN)
        ids = df['STUDY_ID'].unique()
        idsSel = ids[lb:ub]
        dfPO2_FiO2 = df['pO2_FiO2']
        dfSelPO2FO2 = dfPO2_FiO2[df['STUDY_ID'] == ids[0]

                                 ]
        # ==============================================
        Q, P, Pcp = offcd.offline_changepoint_detection(dfSelPO2FO2, partial(offcd.const_prior, l=(len(df) + 1)),
                                                        offcd.gaussian_obs_log_likelihood, truncate=-40)
        print(Q)
        print(P)
        print(Pcp)
        # ================================

        for idSel in idsSel:
            dfSelPO2FO2 = dfPO2_FiO2[df['STUDY_ID'] == idSel]
            fig, ax = plt.subplots(figsize=[18, 16])
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(dfSelPO2FO2)
            ax = fig.add_subplot(2, 1, 2, sharex=ax)
            ax.plot(np.exp(Pcp).sum(0))
            plt.show()
            figPath2Save = Path(outPathDir)/f'changePoint_{idSel}.png'
            fig.savefig(figPath2Save)

        print('Finished')

if __name__ == "__main__":
    main()
