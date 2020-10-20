import pandas as pd
import numpy as np

import sys
import os
# Append the library path to PYTHONPATH, so library can be imported.
sys.path.append(os.path.dirname(os.getcwd()))

from library import common as cm
from library import loader_aux as laux


import setup



if sys.argv[1] == 'tune':
    print('Loading Hyper Tuning data sets!\n')
    data_dir = setup.DATA_DIR + f'CleanData/CONFIG={setup.CONFIG}/TuneHypers/'
    mc_dir = data_dir + 'MC/'


if sys.argv[1] == 'normal':
    print('Loading Normal data sets!\n')
    data_dir = setup.DATA_DIR + f'CleanData/CONFIG={setup.CONFIG}/'
    mc_dir = data_dir + 'MC/'


df = pd.read_csv(data_dir + 'train_val_sim.csv', index_col=0, parse_dates=['date'])
ori_size = df.shape[0]


"""
Remove all extra cols for other hedging periods, except one.
"""
laux.remove_cols_rename(df, setup.OFFSET_DICT, setup.FREQ)
df['on_ret'] = np.exp(df['short_rate'] * setup.DT)


"""
Now, we tag train, validation and test sets.
"""
laux.tag_data(
    df, tag=0, period=0, 
    offset=setup.OFFSET_DICT[setup.FREQ][0], 
    start_date=setup.UNDERLYINGPARAS['start_date'], 
    end_date=setup.DATE_BREAK)
laux.tag_data(
    df, tag=1, period=0,
    offset=setup.OFFSET_DICT[setup.FREQ][0],
    start_date=setup.DATE_BREAK,
    end_date=setup.UNDERLYINGPARAS['end_date']
)


print("Load and clean the training and validation data.")
print(f'Original data size is {df.shape[0]}')


# Remove certain types of samples from training and validation sets.

df = laux.choose_half_shrink_moneyness(df, ori_size, setup.HALF_MONEY, setup.MIN_M, setup.MAX_M)
bl = df['V1'].notna()
cm.print_removal(df.shape[0], sum(bl), ori_size, 'We remove samples when S1 is not available')

df_train = df.loc[bl]
df_train = laux.make_features(df_train)
del df


print("\n\n====================")
print("Clean and load all Monte Carlo test data.\n")


# Import all the monte carlo sets together, and do the same selection for safty reason. Do not select any data in other part of the code.

mc_sets = []    
for i in range(setup.NUM_TEST):
    print('Load Monte Carlo set', i+1)
    df_test = pd.read_csv(mc_dir + 'mc{}.csv'.format(i), index_col=0, parse_dates=['date'])
    
    laux.remove_cols_rename(df_test, setup.OFFSET_DICT, setup.FREQ)
    df_test['on_ret'] = np.exp(df_test['short_rate'] * setup.DT)
    
    laux.tag_data(
        df_test, tag=2, period=0,
        offset=setup.OFFSET_DICT[setup.FREQ][0],
        start_date=df_test['date'].min(),
        end_date=df_test['date'].max()
    )
    ori_size = df_test.shape[0]
    
   
    
    
    df_test = laux.choose_half_shrink_moneyness(df_test, ori_size, setup.HALF_MONEY, setup.MIN_M, setup.MAX_M)
    bl = df_test['V1'].notna()
    cm.print_removal(df_test.shape[0], sum(bl), ori_size, 'We remove samples when S1 is not available')
    df_test = df_test.loc[bl]
    df_test = laux.make_features(df_test)
    mc_sets.append(df_test.copy())
    print('\n')
