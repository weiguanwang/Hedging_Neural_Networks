import datetime

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy.optimize import brentq

from . import heston as hs


def get_time2maturity(
        dates,
        maturities):
    """
    this method returns time to maturity in year fraction.
    Both dates and maturities need to be lists.
    We set maturity to be beginning of day.
    """
    day_count = 253
    mat = pd.DatetimeIndex(maturities)
    tau = np.busday_count(
        dates.values.astype('datetime64[D]'),
        mat.values.astype('datetime64[D]')
    )
    return tau / day_count


def interp_rate2expiry(group, num_days):
    # obtains interest rate to expiry by interpolation
    ds = group['days'].values
    rs = group['rate'].values
    if len(ds) > 1:
        func = interp1d(
            ds, rs, bounds_error=False, fill_value=np.nan)
        var = pd.DataFrame(
            {'days': num_days,
            'rate': func(num_days)})
        return var
    else:
        var = pd.DataFrame(
            {'days': num_days,
            'rate': np.nan})
        return var


# used for simulation and SP500 data
def add_tomorrow(
        df,
        offset_bday,
        offset_key='_1D',
        future_vol=None
    ):
    """
    this methods puts relevant tomorrow's information into dataframe  
    Parameters:
    ============================
    options:
        df: dataframe
    offset_bday:
        this argument the gap between current price and next price.
        1 for daily 5 for weekly. Business day.
    future_vol:
        future trading volume to be joined.
    Return:
    ============================
    dataframe, 2-D DF.
    """
    tmp = ['date', 'optionid', 'S0', 'V0', 'implvol0']
    new_cols = {
        'S0': f'S{offset_key}', 
        'V0': f'V{offset_key}', 
        'implvol0': f'implvol{offset_key}'}
    if 'Var0' in df.columns:
        tmp += ['Var0']
        new_cols['Var0'] = f'Var{offset_key}'

    if future_vol:
        tmp += ['volume']
        new_cols['volume'] = f'volume{offset_key}'
        
    df_tmp = df[tmp].copy()
    df_tmp['date'] -= offset_bday

    df_tmp.rename(columns=new_cols, inplace=True) 

    df = df.join(df_tmp.set_index(['date', 'optionid']), on=['date', 'optionid'])

    return df



def bs_formula_call(vol, S, K, tau, r):
    """
    This is the scalar version of BS, used for element-wise calculating
    implied volatility.
    """
    d1 = ((np.log(S/K) + (r + vol ** 2 / 2) * tau)
            / (np.sqrt(vol ** 2 * tau)))
    d2 = d1 - np.sqrt(vol**2 * tau)
    C = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2) 
    return C


def bs_formula_put(vol, S, K, tau, r):
    """
    This is the scalar version of BS, used for element-wise calculating
    implied volatility.
    """
    d1 = ((np.log(S/K) + (r + vol ** 2 / 2) * tau)
            / (np.sqrt(vol ** 2 * tau)))
    d2 = d1 - np.sqrt(vol**2 * tau)
    P = norm.cdf(-d2) * K * np.exp(-r * tau) - norm.cdf(-d1) * S
    return P


def bs_call_price(vol, S, K, tau, r):
    """
    Vectorized version of BS
    """
    ix_int = (tau < 1e-6)
    C = np.zeros((len(S)))
    C[ix_int] = np.maximum(0, (S - K))[ix_int]

    C[~ix_int] = bs_formula_call(vol[~ix_int], S[~ix_int], K[~ix_int], tau[~ix_int], r[~ix_int])

    return C


def calc_d1(vol, S, K, tau, r):
    ix_int = (tau < 1e-6)
    d1 = np.zeros(len(S))

    d1[~ix_int] = (np.log(S[~ix_int] / K[~ix_int]) + (r[~ix_int] + vol[~ix_int] ** 2 / 2) * tau[~ix_int]) / np.sqrt(vol[~ix_int] ** 2 * tau[~ix_int])
    d1[ix_int] = (np.log(S[ix_int] / K[ix_int]) + (r[ix_int] + vol[ix_int] ** 2 / 2) * tau[ix_int]) / np.sqrt(vol[ix_int] ** 2 * (tau[ix_int] + 1e-10))
    
    return d1


def bs_call_delta(vol, S, K, tau, r):
    delta = norm.cdf(calc_d1(vol, S, K, tau, r))
    return delta


def bs_put_delta(vol, S, K, tau, r):
    return bs_call_delta(vol, S, K, tau, r) - 1.


def bs_call_theta(vol, S, K, tau, r):
    d1 = calc_d1(vol, S, K, tau, r)
    d2 = d1 - vol * np.sqrt(tau)
    theta = - S * norm.pdf(d1) * vol / (2 * np.sqrt(tau + 1e-10)) - r * K * np.exp(-r * tau) * norm.cdf(d2)
    return theta


def bs_gamma(vol, S, K, tau, r):
    """
    Calls and puts have same gamma.
    """

    d1 = calc_d1(vol, S, K, tau, r)
    ix_int = (tau < 1e-6)
    gamma = np.zeros(len(S))
    gamma[ix_int] = norm.pdf(d1[ix_int]) / (S[ix_int] * np.sqrt(tau[ix_int] + 1e-10) * vol[ix_int])
    gamma[~ix_int] = norm.pdf(d1[~ix_int]) / (S[~ix_int] * np.sqrt(tau[~ix_int]) * vol[~ix_int])

    return gamma


def bs_vega(vol, S, K, tau, r):
    d1 = calc_d1(vol, S, K, tau, r)
    vega = S * norm.pdf(d1) * np.sqrt(tau)

    return vega


def bs_vanna(vol, S, K, tau, r):
    """
    Calculate the derivative of delta with resepct to vol
    """
    d1 = calc_d1(vol, S, K, tau, r)
    d2 = d1 - vol * np.sqrt(tau)

    vanna = -1 * norm.pdf(d1) * d2 / vol

    return vanna


def calc_implvol_core(
        given_price, S, K,
        tau, r, bs_formula):
    """
    Calculate implied volatility.
    :param myprice: given_price for the option.
    :param bs_formula: this can be a call pricing formula or
        a put pricing formula.
    """
    try:
        vol = brentq(
            lambda x: given_price - bs_formula(vol=x, S=S, K=K, tau=tau, r=r),
            0.0001, 1000.)
    except ValueError:
        vol = np.nan
    return vol


def find_impvol(row):
    s, k = row['S0'], row['K']
    tau, myprice = row['tau0'], row['V0']
    r = row['r']
    if row['cp_int'] == 0:
        bs_form = bs_formula_call
    if row['cp_int'] == 1:
        bs_form = bs_formula_put

    vol = calc_implvol_core(
        given_price=myprice,
        S=s,
        K=k,
        tau=tau,
        r=r,
        bs_formula=bs_form
    )
    return vol


def calc_implvol(df):   #wrapper
    ts = datetime.datetime.now()
    print('Start time (computing impl vol):', ts)
    df.loc[:, 'implvol0'] = df.apply(find_impvol, axis=1)
    te = datetime.datetime.now()
    print('Finish time (computing impl vol):', te)
    print('Time spent: ', te - ts)
    return df


def normalize_prices(
        df,
        s_divisor,
        norm_factor,
        cols):
    cols_after = [name + '_n' for name in cols]
    df = df.reindex(
        columns=df.columns.tolist() + cols_after
    )
    df[cols_after] = df[cols].values / (s_divisor / norm_factor)[:, np.newaxis]
    return df


def append_1M_ATM_option(mc_one, paras):
    """ Append 1 month ATM option to each date, and its future prices and greeks """
    underlying_model = paras['underlying_model']
    underlying_paras = paras['underlying_paras']
    other_paras = paras['other_paras']
    offset_dict = paras['offset_dict']
    if underlying_model == 'Heston':
        heston_pricer = hs.ComputeHeston(
            underlying_paras['kappa'], 
            underlying_paras['theta'], 
            underlying_paras['sigma'], 
            underlying_paras['rho'], 
            other_paras['short_rate'])
        
    use_cols = ['date', 'S0', 'short_rate', 'r']
    if underlying_model == 'Heston':
        use_cols += ['Var0']
    mc_atm = mc_one[use_cols]
    " We only need one row for each day "
    mc_atm = mc_atm.drop_duplicates()
    " ATM and one-month "
    mc_atm['K'] = mc_atm['S0']
    mc_atm['tau0'] = 1/12.
    " Assume all CALLs "
    mc_atm['cp_int'] = 0
    " Calculate V0 "
    if underlying_model == 'BS':
        mc_atm['implvol0'] = underlying_paras['volatility']
        mc_atm['V0'] = bs_call_price(mc_atm['implvol0'], mc_atm['S0'], mc_atm['K'], mc_atm['tau0'], mc_atm['r'])

    elif underlying_model == 'Heston':
        mc_atm = hs.hs_price_1M_ATM_wrapper(mc_atm, heston_pricer, s_name=f'S0',
                                       var_name=f'Var0', tau_name=f'tau0',
                                       v_name=f'V0')   
        
        mc_atm = calc_implvol(mc_atm)
    mc_atm.rename({'V0': 'V0_atm'}, inplace=True, axis=1)
        
    " For each hedging period "
    for key, value in offset_dict.items():
        tmp = ['date', 'S0']
        new_cols = {
                'S0': f'S{value[1]}'}
        if underlying_model == 'Heston':
            tmp += ['Var0']
            new_cols['Var0'] = f'Var{value[1]}'

        df_tmp = mc_atm[tmp].copy()
        df_tmp['date'] -= value[0]
        df_tmp.rename(columns=new_cols, inplace=True) 
        " Join S and Var by date with offset "
        mc_atm = mc_atm.join(df_tmp.set_index(['date']), on=['date'])
        mc_atm[f'tau{value[1]}'] = mc_atm['tau0'] - value[0].n / 253.
        
        if underlying_model == 'BS':
            mc_atm[f'V{value[1]}_atm'] = bs_call_price(
                mc_atm['implvol0'], mc_atm[f'S{value[1]}'], mc_atm['K'], 
                mc_atm[f'tau{value[1]}'], mc_atm['r'])
        elif underlying_model == 'Heston':
            " Calculate V_1D_atm, V_2D_atm, V_5D_atm"
            " No need to calculate implvol for the future "
            mc_atm = hs.hs_price_1M_ATM_wrapper(mc_atm, heston_pricer, s_name=f'S{value[1]}',
                                       var_name=f'Var{value[1]}', tau_name=f'tau{value[1]}',
                                       v_name=f'V{value[1]}_atm')
    
    """ Normalize price and Calculate Sensitivity """
    cols_to_normalize = (['S' + value[1] for key, value in offset_dict.items()]
                         + [f'V{value[1]}_atm' for key, value in offset_dict.items()])
    mc_atm = normalize_prices(
        mc_atm,
        s_divisor=mc_atm['S0'],
        norm_factor=other_paras['norm_factor'],
        cols=['S0', 'V0_atm', 'K'] + cols_to_normalize
    )
    
    vol, S, K, tau, r = mc_atm['implvol0'], mc_atm['S0_n'], mc_atm['K_n'], mc_atm['tau0'], mc_atm['r']
    mc_atm['delta_bs_atm'] = bs_call_delta(vol, S, K, tau, r) 
    mc_atm['vega_atm_n'] = bs_vega(vol, S, K, tau, r )
    mc_atm['gamma_atm_n'] = bs_gamma(vol, S, K, tau, r )
    mc_atm['vanna_atm_n'] = bs_vanna(vol, S, K, tau, r )
    mc_atm['theta_atm_n'] = bs_call_theta(vol, S, K, tau, r)

    if underlying_model == 'Heston':
        " Calculate Heston Delta and Vega "
        mc_atm = hs.calc_Heston_delta_vega_wrapper(
            mc_atm, heston_pricer, 'S0_n', 'K_n', 'Var0', 'tau0', 'delta_hs_atm', 'vega_hs_atm_n'
        )
        

    """ Join by date """
    use_cols = ['date', 'V0_atm', 'V_1D_atm', 'V_2D_atm', 'V_5D_atm', 'V0_atm_n', 'V_1D_atm_n',
          'V_2D_atm_n', 'V_5D_atm_n', 'delta_bs_atm', 'vega_atm_n', 'gamma_atm_n',
           'vanna_atm_n', 'theta_atm_n']
    if underlying_model == 'Heston':
        use_cols += ['delta_hs_atm', 'vega_hs_atm_n']
        
    mc_one = mc_one.join(mc_atm[use_cols].set_index(['date']), on=['date'])

    return mc_one