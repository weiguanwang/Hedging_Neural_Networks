import getpass
import sys
import os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import datetime


if getpass.getuser() in ['Weiguan', 'weiguan']:
    if sys.platform == 'win32':
        DATA_DIR = 'C:\\Users\\Weiguan\\Dropbox\\Research\\DeepHedging\\Data\\Euroxx\\'
    if sys.platform == 'linux':
        DATA_DIR = '/home/weiguan/Dropbox/Research/DeepHedging/Data/Euroxx/'
if getpass.getuser() in ['rufj']:
    DATA_DIR = '/Users/rufj/Desktop/Weiguan/Weiguan Data/Euroxx/'

os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_SEED = 600

NORM_FACTOR = 100

# Matching delay
MAX_LAG = pd.Timedelta('1 hours')
MATCH_TOL_DICT = {
    '1H': pd.Timedelta('1 hours'),
    '0.5H': pd.Timedelta('0.5 hours'),
    '0.1H': pd.Timedelta('0.1 hours')
}
MATCH_TOL = '0.1H'

# Hedging frequency
OFFSET_DICT = {
    '1H': [pd.Timedelta('1 hours'), '_1H'],
    '1D': [BDay(1), '_1D'],
    '2D': [BDay(2), '_2D']
}


VIXPARAS = {
    'vix0': 13,
    'kappa': 1,
    'sigma': 25,
    'mu': 15
}

FREQ = '1D'

T_FIRSTHALF = datetime.time(8, 30)
if FREQ == '1H':
    DT = 0.
    T_LASTHALF = datetime.time(15, 15)
if FREQ == '1D':
    DT = 1. / 253.
    T_LASTHALF = datetime.time(16, 0)
if FREQ == '2D':
    DT = 2. / 253.
    T_LASTHALF = datetime.time(16, 0)
    
# Rolling window setup
DATE_BEGIN = pd.Timestamp('2016-01-04')
DATE_END = pd.Timestamp('2018-07-26')

# """ Rolling window """
# span_train = '360D'
# span_val = '90D'
# span_test = '90D'
# date_window = '90D'

""" Single window """
span_train = '600D'
span_val = '150D'
span_test = '150D'
date_window = '150D'


SPAN_TRAIN = pd.Timedelta(span_train)
SPAN_VAL= pd.Timedelta(span_val)
SPAN_TEST = pd.Timedelta(span_test)
DATE_WINDOW = pd.Timedelta(date_window)

# Liquidity. MAX_LAG_LIQ is the maximum tolerance when cleaning the data.
# LAG_LIQ_TOL is the tolerance when loading the data, which must be smaller than MAX_LAG_LIQ.
MAX_LAG_LIQ = pd.Timedelta('2 hours')
LAG_LIQ_TOL_f = 0.25
LAG_LIQ_TOL = pd.Timedelta(f'{LAG_LIQ_TOL_f} hours')

# 'otm' means out-of-money only, 'itm' in the money only, 'both' keep both half
HALF_MONEY = 'otm'

# Some additional selections after data are cleaned.
MIN_M, MAX_M = 0.8, 1.5
AGG_SIDE_FLAG = False


PERMUTE = False
NUM_PERMUTE = 5

VIX = False


""" Closeness gap """
CLOSENESS = False
CLOSENESS_GAP = pd.Timedelta('5m')


# business day
MIN_TAU_Days = 0
MIN_TAU = MIN_TAU_Days / 253.

"""
Below is network setup.
"""
# Output activation function.
# OUTACT = 'normcdf'
OUTACT = 'linear'

# Network feature choice
#FEATURE_SET = 'normal_feature'
#FEATURE_SET = 'delta_vega'
FEATURE_SET = 'delta_vega_vanna'
#FEATURE_SET = 'spot_strike'
#FEATURE_SET = 'spot_strike_2'


# Tune Hyperparameters
NUM_REPEATS = 5
END_PERIODS = 1


"""
Set result folder
"""
res_dir = f'{DATA_DIR}Result/FREQ={FREQ}_HALFMONEY={HALF_MONEY}_MINM={MIN_M}_MAXM={MAX_M}_MINTAU={MIN_TAU_Days}_Permute={PERMUTE}_VIX={VIX}_WINDOW={date_window}_AGGSIDE={AGG_SIDE_FLAG}_MATCHING={MATCH_TOL}_CLOSENESS={CLOSENESS}/'