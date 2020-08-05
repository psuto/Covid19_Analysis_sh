import pytest
import utilities as u
from gps.preproces_Covid_gps import ProcessingCovid4GPsContv01

dirPath = r"C:\work\dev\dECMT_src\data_all\COVID19_Data\Current"

@pytest.fixture()
def dfs00():
    filesDict = u.CovidDataPreProcessing.filesDict
    # eval('a'+'=None')
    dfs = u.readFiles(dirPath, filesDict)
    return dfs

@pytest.fixture
def dfs01_datesConverted(dfs00):
    dates2Convert = u.CovidDataPreProcessing.dates2Convert
    # convertColumns4Tables2Datetime
    dfs03 = u.CovidDataPreProcessing.convertColumns4Tables2Datetime(dfs00,dates2Convert)
    return dfs03


def test_convert_columns2datetime(dfs01_datesConverted):
    ProcessingCovid4GPsContv01.convertDates()
    assert False
