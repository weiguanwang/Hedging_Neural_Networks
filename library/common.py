import itertools
import os
import copy

import numpy as np
import pandas as pd

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
    

def permute_core(df, i_period, random_seed):
    """
    df: is a union of training, validation (if any), and the test.
    """ 
    df_train = df.loc[(df[f'period{i_period}'] == 0)]
    df_val = df.loc[(df[f'period{i_period}'] == 1)]
    df_test = df.loc[(df[f'period{i_period}'] == 2)]
    val_size = df_val.shape[0]
    test_size = df_test.shape[0]
    
    id_total = df_train.index.append([df_val.index, df_test.index])
    _, new_test = train_test_split(id_total, test_size=test_size, random_state=random_seed)
    new_train, new_val = train_test_split(_, test_size=val_size, random_state=random_seed)
    
    # We permute simply by changing flag value. 
    df.loc[new_train, f'period{i_period}'] = 0
    df.loc[new_val, f'period{i_period}'] = 1
    df.loc[new_test, f'period{i_period}'] = 2 
    df.loc[new_test, 'Is_In_Some_Test'] = True
    return df


def rolling_permute(df, random_seed):
    """
    This function calls the permute_core and permute for each period.
    """
    max_period = max([int(s[6:]) for s in df.columns if 'period' in s])
    df['Is_In_Some_Test'] = False
    for i in range(max_period + 1):
        df = permute_core(df, i, random_seed=random_seed+i)
    return df 


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
        
        bs_name = 'Regression/BS_Benchmark'
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