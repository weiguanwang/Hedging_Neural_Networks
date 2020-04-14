import sys
import os
# Append the library path to PYTHONPATH, so library can be imported.
sys.path.append(os.path.dirname(os.getcwd()))
import pandas as pd
import numpy as np

import setup
from library import common as cm
from library import stoxx as st
from library import loader_aux as laux

file_path = setup.DATA_DIR + f'CleanData/options.csv'


time_cols = ['ExecuteTime' + value[1] for key, value in setup.OFFSET_DICT.items()]

df = pd.read_csv(
    file_path,
    index_col=0,
    parse_dates=['ExecuteTime0', 'FuturesExpiry', 'Expiry', 'date'] + time_cols,
    dtype={'SecurityType': 'category',
            'TrdType': 'category',
            'AggressorSide': 'category',
          'cp_int': 'int'})

df = df.reset_index(drop=True)
ori_size = df.shape[0]
df['on_ret'] = np.exp(df['short_rate'] * setup.DT)

""" 
For Stoxx, we don't need Stock price, we only need future price. 
We keep stock price in other columns rather than S0
Furthermore, we need to rename 'FuturesPx..' to 'S...'
"""
tmp = {'S0': 'Stock0', 'S0_n': 'Stock0_n'}
df.rename(columns=tmp, inplace=True)
st.rename_future_to_stock(df)


"""
We first select the hedging offset, and remove all others.
"""
laux.remove_cols_rename(df, setup.OFFSET_DICT, setup.FREQ)
df.rename(columns={f'FuturesID{setup.OFFSET_DICT[setup.FREQ][1]}': 'FuturesID1'}, inplace=True)
"""
We also need to use execute time to change matching offset.
"""
name_map = {f'ExecuteTime{setup.OFFSET_DICT[setup.FREQ][1]}': 'ExecuteTime1'}
df.rename(columns=name_map, inplace=True)



""" Remove samples within one hedging offset of option expiry """
bl = (df['ExecuteTime0'] + setup.OFFSET_DICT[setup.FREQ][0]) < df['Expiry']
cm.print_removal(df.shape[0], sum(bl), ori_size, 'We remove samples within one hedging offset before OPTION expiry')
df = df.loc[bl]



"""
We need to drop quotes where the one-step-ahead prices are not available.
"""
bl = df['V1'].notna()
cm.print_removal(df.shape[0], sum(bl), ori_size, 'We remove samples when the next trade is not available')
df = df.loc[bl]

"""  we clean according to matching delay tolerance """
bl = (df['ExecuteTime1'] - (df['ExecuteTime0'] + setup.OFFSET_DICT[setup.FREQ][0])) <= setup.MATCH_TOL_DICT[setup.MATCH_TOL]
cm.print_removal(df.shape[0], sum(bl), ori_size, f'We remove samples when the matching tol is larger than {setup.MATCH_TOL}')
df = df.loc[bl]

"""
If we still have NAs for implied volatility at this point of time, we drop them. 
"""
bl = df['implvol0'].isna()
if sum(bl) > 0.5:
    print('Check Why implvol is not available!\n\n')
bl = df['implvol0'].notna()
cm.print_removal(df.shape[0], sum(bl), ori_size, 'We remove NA implvol0')
df = df.loc[bl]


bl = (df['implvol0'] <= 1.) & (df['implvol0'] >= 0.01)
cm.print_removal(df.shape[0], sum(bl), ori_size, 'We remove unreasonable implvol0')
df = df.loc[bl]


""" Remove first and last half an hour in each trading day """
bl = df['ExecuteTime0'].apply(
    lambda x: (x.time() < setup.T_LASTHALF) and (x.time() > setup.T_FIRSTHALF))
cm.print_removal(df.shape[0], sum(bl), ori_size, 'We remove the first and the last trading period of each day')
df = df.loc[bl]



""" Remove samples near future expiry """
bl = (df['ExecuteTime0'] + setup.OFFSET_DICT[setup.FREQ][0]) < df['FuturesExpiry']
cm.print_removal(df.shape[0], sum(bl), ori_size, 'We remove samples within one hedging offset before FUTURES expiry')
df = df.loc[bl]



"""
1. We choose out-of-money calls and puts only.
2. We further restrict the range of moneyness, so that deep out-of-money are excluded.
"""
# To make sure no NAs in columns
assert df['delta_bs'].isna().sum() < 0.5

df = laux.choose_half_shrink_moneyness(df, ori_size, setup.HALF_MONEY, setup.MIN_M, setup.MAX_M)


bl = df['tau0'] > setup.MIN_TAU
cm.print_removal(df.shape[0], sum(bl), ori_size, 
                 f'We remove samples that have time to maturity less than {setup.MIN_TAU_Days} day')
df = df.loc[bl]


"""
extra: remove trades that have consecutive trades for the same option within a threshold
"""
if setup.CLOSENESS:
    grouped = df.groupby(by='SecurityID')
    all_index = []
    for name, group in grouped:
        all_index += st.search_pivot(group, gap=setup.CLOSENESS_GAP, col='ExecuteTime0')
    cm.print_removal(df.shape[0], len(all_index), ori_size, 'We remove trades that are too close for same option')
    df = df.loc[all_index]


"""
Make features for regression and network
"""
df = laux.make_features(df)


df, df_dates = laux.rolling_window(
    df, 
    date_begin=setup.DATE_BEGIN,
    span_train=setup.SPAN_TRAIN,
    span_val=setup.SPAN_VAL,
    span_test=setup.SPAN_TEST,
    date_window=setup.DATE_WINDOW,
    date_end=setup.DATE_END,
    offset=setup.OFFSET_DICT[setup.FREQ][0]
)

