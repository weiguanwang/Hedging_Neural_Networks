import itertools
import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.tseries.offsets import BDay, DateOffset, WeekOfMonth
from sklearn.model_selection import train_test_split


def print_removal(before_size, cur_size, ori_size, issue):
    print(f'{issue}. {before_size - cur_size} samples ({(before_size - cur_size)/before_size* 100:.2f}%) are removed. We have {cur_size / ori_size * 100:.2f}% of original data left, yielding a size of {cur_size}.')


def calc_pnl(
        df, delta,
        V1='V1_n'
):
    """
    This method calculates PnL. We assume short option.
    :param V1: Target to compare to.
    :return: Series pnl
    """
    s0, s1 = df['S0_n'], df['S1_n']
    v0, v1 = df['V0_n'], df[V1]
    on_return = df['on_ret']
    
    v1_hat = (v0 - delta * s0) * on_return + delta * s1
    return (v1_hat - v1)


def store_pnl(
        df, delta,
        pnl_path,
        V1='V1_n'
):
    delta = delta[df['Is_In_Some_Test']]
    df = df[df['Is_In_Some_Test']]

    # Cap or floor PNL by security type
    bl_c = df['cp_int'] == 0
    delta[bl_c] = np.maximum(delta[bl_c], 0.)
    delta[~bl_c] = np.minimum(delta[~bl_c], 0.)

    cols = [x for x in df.columns if x in ['ExecuteTime0', 'Aggressorside']]
    cols += ['cp_int', 'date']
    df_res = df[cols].copy()
    df_res['delta'] = delta
    df_res['PNL'] = calc_pnl(df, delta, V1=V1)
    df_res['M0'] = df['M0'].copy()
    df_res['tau0'] = df['tau0'].copy()
    
    df_res['testperiod'] = np.nan
    # In addition, we want to record which test period the pnl is from.
    max_period = max([int(s[6:]) for s in df.columns if 'period' in s])
    for i in range(0, max_period + 1):
        bl = df['period{}'.format(i)] == 2
        df_res.loc[bl, 'testperiod'] = i
        
    df_res.to_csv(pnl_path)
    

def calc_pnl_two_assets(df, delta, eta, V1='V1_n'):
    """ 
    This methods calcuate the PnL, given a strategy of two 
    hedging instruments.
    For the moment, we use underlying and the ATM one-month option.
    """
    s0, s1 = df['S0_n'], df['S1_n']
    v0, v1 = df['V0_n'], df[V1]
    atm0, atm1 = df['V0_atm_n'], df['V1_atm_n']
    on_return = df['on_ret']

    v1_hat = (v0 - delta * s0 - eta * atm0) * on_return + delta * s1 + eta * atm1
    return v1_hat - v1


def store_pnl_two_assets(df, delta, eta, pnl_path, V1='V1_n'):
    delta = delta[df['Is_In_Some_Test']]
    eta = eta[df['Is_In_Some_Test']]
    df = df[df['Is_In_Some_Test']]

    cols = [x for x in df.columns if x in ['ExecuteTime0', 'Aggressorside']]
    cols += ['cp_int', 'date']
    df_res = df[cols].copy()
    df_res['delta'] = delta
    df_res['eta'] = eta
    df_res['PNL'] = calc_pnl_two_assets(df, delta, eta, V1=V1)
    df_res['M0'] = df['M0'].copy()
    df_res['tau0'] = df['tau0'].copy()
    
    df_res['testperiod'] = np.nan
    # In addition, we want to record which test period the pnl is from.
    max_period = max([int(s[6:]) for s in df.columns if 'period' in s])
    for i in range(0, max_period + 1):
        bl = df['period{}'.format(i)] == 2
        df_res.loc[bl, 'testperiod'] = i
        
    df_res.to_csv(pnl_path)



class Inspector:
    def __init__(self):
        pass

    def loadPnl(self, path, measure, op_type=None):
        df = pd.read_csv(path, index_col=0)


        bl = self.choose_op_type(df, op_type)

        if measure == 'mse':
            return (df.loc[bl, 'PNL']**2).mean()
        elif measure == 'mean':
            return (df.loc[bl, 'PNL']).mean()
        elif measure == 'median':
            return (df.loc[bl, 'PNL']).median()
        elif measure == 'lower5%VaR':
            return (df.loc[bl, 'PNL']).quantile(0.05)
        elif measure == 'upper95%VaR':
            return (df.loc[bl, 'PNL']).quantile(0.95)
        else:
            raise NotImplementedError('The given measure is not implemented!')

            
    def choose_op_type(self, df, op_type):
        if op_type == 'call':
            bl = df['cp_int'] == 0
        elif op_type == 'put':
            bl = df['cp_int'] == 1
        else: 
            bl = df['cp_int'].notna()
        return bl


    def evalPnls(self, df_dirs, aggregating, measure, op_type=None):
        """
        Params:
        =========================
        aggregating: the aggregating metod over all PNL files
        measure: the measure to evaluate on each PNL file.
        """
        rows, cols = df_dirs.index, df_dirs.columns
        sub_cols = ['Absolute', '%Change']
        cols_indices = pd.MultiIndex.from_product([cols, sub_cols], names=['setup', 'value'])
        df_res = pd.DataFrame(index=rows, columns=cols_indices)

        for r, c in list(itertools.product(rows, cols)):
            directory = os.fsencode(df_dirs.loc[r, c] + 'pnl/')
            res = []
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    filename = os.fsdecode(directory + file)
                    if filename.endswith(".csv"):
                        res.append(self.loadPnl(filename, measure, op_type))
                
                if aggregating is 'mean':
                    df_res.loc[r, (c, 'Absolute')] = sum(res) / len(res)
                else:
                    raise NotImplementedError('The given aggregating is not implemented!')
            else:
                df_res.loc[r, c] = np.nan
        
        bs_name = [x for x in df_dirs.index.tolist() if 'BS_Benchmark' in x][0]
        for c in cols:
            tmp = (df_res.loc[:, (c, 'Absolute')] - df_res.loc[bs_name, (c, 'Absolute')]) / \
                df_res.loc[bs_name, (c, 'Absolute')] * 100.
            tmp = tmp.astype(np.float)
            df_res.loc[:, (c, '%Change')] = tmp.round(2)
            df_res.loc[:, (c, 'Absolute')] = (100*df_res.loc[:, (c, 'Absolute')]).astype(np.float).round(3)   #JR 

        return df_res
    

    def eval_single_exp(self, dirs_dict, measure, op_type=None):
        """
        load each PNL file in the `directory` and return a list of measurements.
        """
        df_res = pd.DataFrame()
        for y, x in dirs_dict.items():
            directory = x + '/pnl/'
            res = []
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    filename = os.fsdecode(directory + file)
                    if filename.endswith(".csv"):
                        res.append(self.loadPnl(filename, measure, op_type))
                
                df_res[y] = res
            else:
                df_res[y] = np.nan
        
        return df_res
        


class PnlLoader:
    def __init__(self, dirs_dict=None):
        self.pnl = None
        self.record = pd.DataFrame()
        self.dirs_dict = dirs_dict

    def load_real_pnl(self, idx=None):
        """
        Given a dictionary of paths, return a dictionary of pnl files, 
        with the same keys.
        """
        res = {}
        for name, x in self.dirs_dict.items():
            if idx is None:
                res[name] = f'{x}pnl/pnl.csv'
            else:
                res[name] = f'{x}pnl/pnl{idx}.csv'
        self.pnl = {}
        for key, path in res.items():
            self.pnl[key] = pd.read_csv(path, index_col=0)

    def load_aggregate_simulation(self, num_test):
        self.pnl = {}
        for name, x in self.dirs_dict.items():
            directory = f'{x}pnl/'
            df = pd.DataFrame()
            for i in range(num_test):
                filename = directory + f'pnl{i}.csv'
                df_add = pd.read_csv(filename, index_col=0)
                df_add.drop(columns=['testperiod'], inplace=True)
                # for simulation data, we use `testperiod` to index test sets.
                df_add['testperiod'] = i 
                df = df.append(df_add)
            df = df.reset_index()
            self.pnl[name] = df


#used mostly for plots. summarizes MSHEs for different models
class LocalInspector(PnlLoader):

    def plug_existing(self, pnl):
        self.pnl = pnl

    def choose_op_type(self, df, op_type):
        if op_type == 'call':
            bl = df['cp_int'] == 0
        elif op_type == 'put':
            bl = df['cp_int'] == 1
        else: 
            bl = df['cp_int'].notna()
        return bl

    def compare_period(self, op_type=None):
        """
        In this method, pnl is aggregated for each period.
        """
        for key, pnl in self.pnl.items():
            max_period = int(max(pnl['testperiod']))
            for i in range(max_period + 1):
                bl = pnl['testperiod'] == i
                bl_ = self.choose_op_type(pnl, op_type)
                bl = bl & bl_
                self.record.loc[i, 'num_samples'] = bl.sum()
                self.record.loc[i, key] = (pnl.loc[bl, 'PNL']**2).mean()

        return self.record   


def compare_pair(daily_mshe, first, second, trunc_qs):
    N = daily_mshe.shape[0]
    print('Size of N:', N)
    diff = daily_mshe[first] - daily_mshe[second]
    for q in trunc_qs:
        cap = diff.abs().quantile(q)
        
        truncated_diff = np.maximum(np.minimum(diff, cap), -cap)
        zscore = truncated_diff.mean() / truncated_diff.std() * np.sqrt(N)
        print(f'Mean difference is {truncated_diff.mean()}')
        print(f'Std is {truncated_diff.std()}')
        print(f'Z-score after truncating at {q} is {zscore}')
        
        truncated_diff.plot()
        plt.show()
        truncated_diff.plot(kind='hist', logy=True, bins=100)
        plt.show()


def truncate_daily_mshe(daily_mshe, first, second, q):
    diff = daily_mshe[first] - daily_mshe[second]
    cap = diff.abs().quantile(q)
    truncated_diff = np.maximum(np.minimum(diff, cap), -cap)
    return truncated_diff
        
def get_zscore(daily_mshe, first, second, q):
    N = daily_mshe.shape[0]
    truncated_diff = truncate_daily_mshe(daily_mshe, first, second, q)
    zscore = truncated_diff.mean() / truncated_diff.std() * np.sqrt(N)
    return zscore


def get_z_confidence(daily_mshe, first, second, q):
    N = daily_mshe.shape[0]
    truncated_diff = truncate_daily_mshe(daily_mshe, first, second, q)
    up = truncated_diff.mean() + 2* truncated_diff.std() / np.sqrt(N)
    down = truncated_diff.mean() - 2* truncated_diff.std() / np.sqrt(N)
    return down, up