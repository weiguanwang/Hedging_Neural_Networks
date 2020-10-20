import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from . import common as cm


def make_predictor(df, features, delta_coeff_1=False):
    
    """ 
    If delta_coeff_1 is to be used, the target is different.
    These predictors are used for linear regressions.
    """

    if delta_coeff_1:
        y = df['V1_n'] - df['V0_n'] * df['on_ret'] - df['delta_bs'] * (df['S1_n'] - df['S0_n'] * df['on_ret'])
    else: 
        y = df['V1_n'] - df['V0_n'] * df['on_ret']
    
    preds = df[features].copy()
    preds = preds.multiply(df['S1_n'] - df['S0_n'] * df['on_ret'], axis=0)
    return y, preds


def fit_lin_core(df, features, delta_coeff_1=False):
    """
    Fit a linear regression on the set of features,
    such that (P&L)^2 is minimized
    """
    y, preds = make_predictor(df, features, delta_coeff_1)
    lin = LinearRegression(fit_intercept=False).fit(preds, y)

    y_hat = lin.predict(preds)
    residual = (y - y_hat)
    residual_sum_of_square = (residual ** 2).sum()
    sigma_square_hat = residual_sum_of_square / (preds.shape[0] - preds.shape[1])
    var_beta = (np.linalg.inv(preds.T @ preds) * sigma_square_hat) 
    std = [var_beta[i, i] ** 0.5 for i in range(len(var_beta))]

    return {'regr': lin, 'std': std}


def predicted_linear_delta(
    lin,  df_test, features
):

    df_delta = pd.Series(index=df_test.index)
    delta = lin.predict(df_test.loc[:, features])
    df_delta.loc[:] = delta

    return df_delta


def rolling_lin_reg(
        df, features,
        end_period,
        agg_side=None,  # if true, data get stratified by aggressor_side, too
        delta_coeff_1=False,  
        leverage=False,
        bucket=None
):
    deltas = pd.Series(index=df.index)
    
    cols_by = ['cp_int']
    if agg_side:
        cols_by += ['AggressorSide']
    if bucket:
        cols_by += ['bucket']

    grouped = df.groupby(by=cols_by)

    groupnames = [str(x) for x in grouped.groups.keys()]
    tmp = pd.MultiIndex.from_product([groupnames, features], names=['Group', 'Coef'])
    df_coef, df_fit_std  =  [pd.DataFrame(index=range(end_period+1), columns=tmp) for _ in range(2)]
    if leverage:
        df_leve = pd.DataFrame(index=range(end_period+1), columns=groupnames)
    else:
        df_leve = None

    for i in range(0, end_period + 1):
        for name, group in grouped:                
 
            bl_train = (group[f'period{i}'] == 0) | (group[f'period{i}'] == 1)
            bl_test = (group[f'period{i}'] == 2)

            df_train = group.loc[bl_train].copy()
            df_test = group.loc[bl_test].copy()

            regs = fit_lin_core(df_train, features, delta_coeff_1)

            if leverage:
                var = fit_leverage(df_train)
                df_leve.loc[i, str(name)] = var

            df_coef.loc[i, str(name)] = regs['regr'].coef_
            df_fit_std.loc[i, str(name)] = regs['std']
            deltas.loc[df_test.index] = predicted_linear_delta(regs['regr'], df_test, features)
            
    return {'delta': deltas, 'df_coef': df_coef, 'df_leve': df_leve, 
            'df_fit_std': df_fit_std}




def run_store_lin(
        vix=None, 
        sub_res=None, 
        pnl_path=None, 
        features=None, 
        df=None,
        max_period=None,
        delta_coeff_1=None,  
        agg_side=None,
        leverage=False,
        bucket=None
):
    if vix:
        print('VIX is used!')
        features += ['fake_vix']

    os.makedirs(sub_res, exist_ok=True)

    res_dict = rolling_lin_reg(
        df, 
        features=features,
        end_period=max_period,
        delta_coeff_1=delta_coeff_1,
        agg_side=agg_side,
        leverage=leverage,
        bucket=bucket
    )
    if not delta_coeff_1:    
        cm.store_pnl(df, res_dict['delta'], pnl_path)   
    else:
        cm.store_pnl(df, res_dict['delta'] + df['delta_bs'], pnl_path)

    return {'df_coef': res_dict['df_coef'], 'df_leve': res_dict['df_leve'], 
            'df_fit_std': res_dict['df_fit_std']}


def fit_leverage(df):
    """
    Fit the leverage between stock return and volatility change.
    """

    """ Because implvol1 may still be NA in df, we need to remove """
    bl = df['implvol1'].notna()
  
    df = df.loc[bl]  
    preds = (df['S1_n'] - df['S0_n']).values.reshape(-1, 1)
    y = df['implvol1'] - df['implvol0']
    lin = LinearRegression(fit_intercept=False).fit(
        preds,
        y
    )
    var = (lin.coef_ * df['vega_n'] / df['delta_bs']).mean()

    return var
