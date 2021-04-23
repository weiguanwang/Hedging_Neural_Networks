import getpass
import sys
import os
import numpy as np
import pandas as pd

from pandas.tseries.offsets import BDay

if getpass.getuser() in ['Weiguan', 'weiguan']:
    if sys.platform == 'win32':
        DATA_DIR = 'C:\\Users\\Weiguan\\Dropbox\\Research\\01_DeepHedging\\Data\\OptionMetrics\\'
    if sys.platform == 'linux':
        DATA_DIR = '/home/weiguan/Dropbox/Research/DeepHedging/Data/OptionMetrics/'
if getpass.getuser() in ['rufj']:
    DATA_DIR = '/Users/rufj/Desktop/Weiguan/Weiguan Data/OptionMetrics/'

os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_SEED = 600


""" Dictionary of offsets for data preparation """
OFFSET_DICT = {
    '1D': [BDay(1), '_1D'],
    '2D': [BDay(2), '_2D'],
    '5D': [BDay(5), '_5D']
}


# normalized prices with this factor
NORM_FACTOR = 100.


# Choose hedge time frequency.
# this is the gap between current and next time stamp.
# Choose 1 for daily, 5 for weekly. Business day convention.
FREQ = '1D'
if FREQ == '1D':    
    DT = 1 / 253.
if FREQ == '2D':
    DT = 2 / 253.
if FREQ == '5D':
    DT = 5 / 253.    

"""
Choose how to fill in the missing implied vol and greeks.
"""

# FEED_MISSING = 'replace_all'
FEED_MISSING = 'replace_missing'
# FEED_MISSING = 'remove_missing'


""" Choose to remove samples for which tomorrow's volume is also zero """
REMOVE_TMR_ZERO_VOLUME = False


# Rolling window setup
DATE_BEGIN = pd.Timestamp('2010-01-01')
DATE_END = pd.Timestamp('2019-06-27')

span_train = '720D'
span_val = '180D'
span_test = '180D'
date_window = '180D'

# """ Single window setup """
# span_train = '2280D'
# span_val = '570D'
# span_test = '570D'
# date_window = '570D'

SPAN_TRAIN = pd.Timedelta(span_train)
SPAN_VAL = pd.Timedelta(span_val)
SPAN_TEST = pd.Timedelta(span_test)
DATE_WINDOW = pd.Timedelta(date_window)


""" choose in-the-money or out-of-money, or both. """
# 'otm' means out-of-money only, 'itm' in the money only, 'both' keep both half
HALF_MONEY = 'otm'
MIN_M, MAX_M = 0.8, 1.5


MIN_TAU_Days = 0
MIN_TAU = MIN_TAU_Days / 253.



    
"""
Network related paras
"""

# Output activation function.
# OUTACT = 'normcdf'
OUTACT = 'linear'

# Network feature choice
#FEATURE_SET = 'normal_feature'
FEATURE_SET = 'delta_vega'
# FEATURE_SET = 'delta_vega_vanna'
#FEATURE_SET = 'spot_strike'
#FEATURE_SET = 'spot_strike_2'

# Hyper parameters tuning
NUM_PERIOD_END = 4
NUM_REPEATS = 1 #5


"""
Set result folder
"""
res_dir = f'{DATA_DIR}Result/FREQ={FREQ}_HALFMONEY={HALF_MONEY}_MINM={MIN_M}_MAXM={MAX_M}_MINTAU={MIN_TAU_Days}_MISSING={FEED_MISSING}_WINDOW={date_window}_KickTmrZeroVolume={REMOVE_TMR_ZERO_VOLUME}/'
