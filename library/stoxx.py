import numpy as np
import pandas as pd

from . import common as cm


# find future price for each option trade
def find_px(
        option_file,
        options_dir,
        futures_dir,
        dtype
):
    df_op = pd.read_csv(
        options_dir + option_file,
        parse_dates=['ExecuteTime0', 'Expiry', 'FuturesExpiry'],
        index_col=0,
        dtype=dtype)
    # Option file has been sorted!.
    df_op.reset_index(drop=True, inplace=True)
    # read in the corresponding futures file name
    future_file = option_file[3:]
    df_fu = pd.read_csv(
        futures_dir + future_file,
        parse_dates=['ExecuteTime0', 'Expiry'],
        index_col=0,
        dtype=dtype)
    df_fu.sort_values('ExecuteTime0', inplace=True)
    df_fu.reset_index(drop=True, inplace=True)
    # search the proper future px
    id_int = df_fu['ExecuteTime0'].searchsorted(df_op['ExecuteTime0'])
    # By using this bl, we can remove trades when there exists no
    # future trades before the option trades.
    bl = (id_int != 0)
    # We need price immediately before
    id_int = np.maximum(id_int - 1, 0)
    # since df_fu index has been reset, doesn't matter on iloc or loc.
    px = df_fu.loc[id_int, 'avg_px']
    df_op['FuturesPx0'] = px.values

    cm.print_removal(df_op.shape[0], sum(bl), df_op.shape[0], 'We remove options without a price for the underlying future')  # small bug: we don't use original size here (as third argument) -- however, doesn't matter as nothing is removed
    
    return df_op.loc[bl]


def find_next_trade_by_group(
        group,
        offset,
        max_lag,
        labels):
    """
    This function works for each group, thus it is
    called by the 'groupby.apply'.
    Arguments:
        =======================
    Each of the groups is  a 'df', a Dataframe.
    'Offset' is the hedging frequency.
    'max_lag' is the greatest lag that can be tolerated.
    Return:
        ====================
        The matching next trades after time length 'offset' for the group.

    """
    df_tmp = group.reset_index(drop=True)
    ts_next = df_tmp['ExecuteTime0'] + offset
    # the price after executetime + offset
    idx_next = df_tmp['ExecuteTime0'].searchsorted(ts_next)
    df_next = df_tmp.reindex(index=idx_next)
    df_next.index = group.index
    # note here ts_next and df_next have different indices,
    # So we need .values for ts_next.
    df_next.loc[df_next['ExecuteTime0'] - ts_next.values > max_lag] = np.nan

    return df_next[labels]


def calc_tau_rates(datetime, expiry, rates):
    """
    Calculate time to maturity and corresponding interest rate to use.
    The corresponding interest rate has the same maturity as option expiry.
    This version of calculating time to maturity is different
    from what is in the common module in 2 aspects.
    Here we account for the time (hours offset) with respect to
    market open time.
    Second, we take the end of day as expiry. This is
    to account for hours time.
    :param time: Datetime series
    :param expiry: Datetime series
    :return: time-to-maturity and rates
    """
    c_date = map(lambda x: x.date(), datetime)
    c_date = pd.DatetimeIndex(list(c_date))

    e_date = map(lambda x: x.date(), expiry)
    e_date = pd.DatetimeIndex(list(e_date))
    # Number of calender days
    num_days = (e_date - c_date).days + 1
    tau = np.busday_count(
        c_date.values.astype('datetime64[D]'),
        e_date.values.astype('datetime64[D]'))
    tau = (tau + 1) / 253.
    market_open = map(
        lambda x: x.replace(hour=8, minute=0, second=0), datetime)
    market_open = pd.DatetimeIndex(list(market_open))
    tau -= ((datetime - market_open) / pd.Timedelta(8.5, unit='h') * 1/253)

    # Multiple indexing rate
    r = rates.reindex(
        index=list(zip(c_date, num_days))).values / 100.

    return tau, r


def rename_future_to_stock(df):
    """
    In the stoxx data set, we use futures as hedging intrument.
    In other parts of the code, we use S0, S1 for the price of
    hedging instrument. So we need to rename the future prices to 
    S0, and so on.
    """
    cols = [x for x in df.columns if 'FuturesPx' in x]
    name_map = {}
    for col in cols:
        name_map[col] = 'S' + col[9:] # col[:9] = '_1H...'
    df.rename(columns=name_map, inplace=True)


def search_pivot(group, gap, col='ExecuteTime0'):
    group = group.sort_values(col)
    group = group.reset_index()
    accepted_index = []
    pivot = 0
    pivot_time = group.loc[pivot, col]
    accepted_index.append(group.loc[pivot, 'index'])
    pivot_time = pivot_time + gap
    while (pivot_time <= group[col].max()):
        pivot = group[col].searchsorted(pivot_time)[0]
        pivot_time = group.loc[pivot, col]
        accepted_index.append(group.loc[pivot, 'index'])
        pivot_time = pivot_time + gap
        
    return accepted_index
    
