import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.stats import norm
from pandas.tseries.offsets import BDay

from . import common as cm
from .bs import simulate_geometric_bm
from .heston import simulate_heston


#plots in-sample path and 10 out-of-sample paths for simulation data
def plot_stock_test_prices(
        path,
        underlying_model,
        underlying_params,
        n_ofTestDays,
        date_break,
        index_end_date):
    fig_sim, ax_sim = plt.subplots()
    path['S0'].plot(ax=ax_sim, legend=False)
    underlying_params['s0'] = path['S0'].iloc[-1]
    underlying_params['start_date'] = underlying_params['end_date'] + BDay()
    underlying_params['end_date'] = underlying_params['end_date'] + n_ofTestDays

    if underlying_model == 'BS':
        for i in range(10):
            simulate_geometric_bm(underlying_params).plot(
                ax=ax_sim,
                legend=False,
                alpha=0.5
            )
    elif underlying_model == 'Heston':
        for i in range(10):
            simulate_heston(underlying_params)['S0'].plot(
                ax=ax_sim,
                legend=False,
                alpha=0.5
            )
        

    ax_sim.annotate(
        'Training',
        xy=(0.4, 0.05),
        xytext=(0.4, 0),
        xycoords='axes fraction',
        ha='center',
        va='bottom')
    #  arrowprops=dict(arrowstyle='-[, widthB=11.0, lengthB=.5', lw=1.5)
    ax_sim.annotate(
        'Validation',
        xy=(0.75, 0.05),
        xytext=(0.75, 0),
        xycoords='axes fraction',
        ha='center',
        va='bottom')
    ax_sim.annotate(
        'Test',
        xy=(0.91, 0.05),
        xytext=(0.91, 0),
        xycoords='axes fraction',
        ha='center',
        va='bottom')
    ax_sim.axvline(index_end_date, color='black', linestyle='dashed', alpha=0.6)
    ax_sim.axvline(date_break, color='black', linestyle='dashed', alpha=0.6)

        
class Painter(cm.PnlLoader):

    def load_period_tables(self):
        self.df_period = pd.DataFrame()
        for name, single_dir in self.dirs_dict.items():
            single = pd.read_csv(f'{single_dir}period_pnl.csv', index_col=0)
            self.df_period[f'{name}'] = single['MSHE']
        return self.df_period.copy()
    
    
    def plot_coef_err(self, model_name, groups, feature):
        """
        groups is a dictionary, the keys are the lables that show for the legend, and
        the value is the group name in csv file.
        """
        df_coef = pd.read_csv(f'{self.dirs_dict[model_name]}coef.csv', header=[0, 1])
        df_std = pd.read_csv(f'{self.dirs_dict[model_name]}std.csv', header=[0, 1])
        for n, g in groups.items():
            plt.errorbar(df_coef.index + 1, df_coef.loc[:, (g, feature)], fmt='none',
                yerr=2*df_std.loc[:, (g, feature)], label=n, capsize=5)
        plt.legend(frameon=False)

    
    # One of the panels in the d1-d1 plot
    def plot_relative_core(self, x1, x2, y, idx, ax):
        c = (np.absolute(self.pnl[x1].loc[idx, 'PNL']) 
            - np.absolute(self.pnl[x2].loc[idx, 'PNL'])) > 0.
        if y == 'd1':
            tmp1 = self.pnl[x1].loc[idx].copy()
            tmp2 = self.pnl[x2].loc[idx].copy()
            
            put_bl = tmp1['cp_int'] == 1
            tmp1.loc[put_bl, 'delta'] += 1
            tmp2.loc[put_bl, 'delta'] += 1

            tmp1 = norm.ppf(tmp1['delta'])
            tmp2 = norm.ppf(tmp2['delta'])

        elif y =='delta':
            tmp1 = self.pnl[x1].loc[idx, 'delta']
            tmp2 = self.pnl[x2].loc[idx, 'delta']
        true_color = sns.color_palette("RdBu_r", 4)[0]
        false_color = sns.color_palette("RdBu_r", 4)[-1]
        c = [true_color if x else false_color for x in c]
        ax.scatter(tmp1, tmp2, s=0.3,  c=c, alpha=0.5)

        
    """ PNL squared against variable ('feature') """
    def pnl_vs_feature_v2(self, df,
            model_name, feature, num_periods, sim_data,
            normalized=True, qcut=True, xlog=False, ylog=False,
            overall=False, xlims=None, ylims=None):
        """
        df: the original loaded clean data.
        """
        if overall: 
            num_periods = 1
        pnl_tmp = self.pnl[model_name]

        """ Remove some in the money sample """
        bl = ((pnl_tmp['cp_int'] == 0) & (pnl_tmp['M0'] < 0.999)) | ((pnl_tmp['cp_int'] == 1) & (pnl_tmp['M0'] > 1.001))
        pnl_tmp = pnl_tmp[bl].copy()

        if feature not in self.pnl[model_name].columns:
            if sim_data: 
                pnl_tmp = pnl_tmp.join(df.set_index(['index', 'testperiod'])[feature], on=['index', 'testperiod'])
            else:
                pnl_tmp = pnl_tmp.join(df[[feature]])
          
        if normalized: 
            if sim_data:
                pnl_tmp = pnl_tmp.join(df.set_index(['index', 'testperiod'])['V0_n'], on=['index', 'testperiod'])
            else:
                pnl_tmp = pnl_tmp.join(df[['V0_n']])
            
            pnl_tmp['pnl_n'] = pnl_tmp['PNL'] / (pnl_tmp['V0_n'])
        else: 
            pnl_tmp['pnl_n'] = pnl_tmp['PNL']
        
        pnl_tmp['PNL_sq'] = pnl_tmp['pnl_n']**2

        fig, axes = plt.subplots(ncols=2, nrows=np.ceil(num_periods/2).astype('int'), 
                                figsize=(16, 4 * np.ceil(num_periods/2)))
        
        fig.subplots_adjust(hspace=0.4)
        axes = axes.flatten()

        for j in range(num_periods):
            ax = axes[j]
            if overall:
                pnl_2 = pnl_tmp.copy()
            else:
                idx = pnl_tmp['testperiod'] == j
                pnl_2 = pnl_tmp.loc[idx].copy()
            if qcut:
                pnl_2[f'{feature}_cat'] = pd.qcut(pnl_2[feature], 10)
            
            tticks = []
            for i, fmt, k in zip([0, 1], ['o', '^'], ['Call', 'Put']):
                pnl_single = pnl_2.loc[pnl_2['cp_int'] == i]
                pnl_sq_group = pnl_single.groupby(f'{feature}_cat')[['PNL_sq', f'{feature}']]
                # dropna because some intervals do not have call or put
                pnl_sq_mean = pnl_sq_group.mean()['PNL_sq'].dropna() 
                
                pnl_sq_std = (pnl_sq_group.std()['PNL_sq'] / np.sqrt(pnl_sq_group.size())).dropna()
                ticks = pnl_sq_group.mean()[f'{feature}'].dropna()
                
                tticks += list(ticks.round(2))      
                ax.errorbar(ticks.values, pnl_sq_mean.values, yerr=2*pnl_sq_std.values, fmt=fmt, capsize=3,label=k)

            if xlog:
                ax.set_xscale('log')
            if ylog:
                ax.set_yscale('log')
            ax.set_title(f'Window {j+1}')
            a = list(set(tticks))
            a.sort()
            ax.set_xticks(a)
            ax.set_xticklabels(a, rotation=90)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.legend()
        return axes



