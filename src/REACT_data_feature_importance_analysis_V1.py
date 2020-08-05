# Zili
import csv
import numpy as np
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
import pickle
import pandas as pd
#import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
import math



5
data_dir='data/'

#table='REACT_COVID_Demographics'
#table='REACT_COVID_Demographics_20200506'
#table='REACT_Events'
#table='REACT_LabResults'
#table='REACT_PharmacyData'
#table='REACT_UHSCOVIDTest_processed'
#table='REACT_Vitalsigns_Categorical'
#table='REACT_Vitalsigns_Numeric'


# Extract AIR_ventilator information from table REACT_Events

air_ventilator_degree=[
    'Air - Not Supported',
    'Nasal Specs',
    'Face Mask',
    'Venturi Mask',
    'Trachy Mask',
    'Non-Rebreath Mask',
    'Optiflow / Hi Flow',
    'NIV - CPAP nasal mask',
    'NIV - CPAP face mask',
    'NIV - CPAP full face mask',
    'NIV - BIPAP nasal mask',
    'NIV - BIPAP face mask',
    'NIV - BIPAP full face mask',
    'Invasive Ventilation'
]
air_ventilator_degree_dic=dict([(air_ventilator_degree[i],float(i)) for i in range(len(air_ventilator_degree))])
air_ventilator_degree_dic
