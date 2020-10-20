import numpy as np
import pandas as pd

from .common import print_removal


def choose_half_shrink_moneyness(df, ori_size, half, min_m, max_m):
    """ Choose in-the-money, out-of-money or both """
    if half == 'otm':
        df = out_of_money_only(df, ori_size)
    elif half == 'itm':
        df = in_the_money_only(df, ori_size)
    elif half == 'both':
        pass
    else:
        raise NotImplementedError('In-the-money or out-of-money')

    bl = (df['M0'] >= min_m-0.001) & (df['M0'] <= max_m+0.001)
    print_removal(df.shape[0], sum(bl), ori_size, 'We shrink moneyness range')
    df = df.loc[bl]
    
    return df


def out_of_money_only(df, ori_size):
    bl = ((df['cp_int'] == 0) & (df['M0'] < 1.001)) | ((df['cp_int'] == 1) & (df['M0'] > 0.999))
    print_removal(df.shape[0], sum(bl), ori_size, 'We remove in-the-money samples')
    df = df.loc[bl]
    return df


def in_the_money_only(df, ori_size):
    bl = ((df['cp_int'] == 0) & (df['M0'] > 0.999)) | ((df['cp_int'] == 1) & (df['M0'] < 1.001))
    print_removal(df.shape[0], sum(bl), ori_size, 'We remove out-of-money samples')
    df = df.loc[bl]
    return df


# Removing columns from dataframe that have to do with specific hedging period (given by single-key)
# Also renames; e.g. future normalized underlying price now denoted by S1_n
# used when loading the data
def remove_cols_rename(df, whole_dict, single_key, future_volume=None):
    remove_freq = whole_dict.copy()
    remove_freq.pop(single_key)

    for key, value in remove_freq.items():
        tag = value[1]
        remove_cols = [x for x in df.columns if tag in x]
        for col in remove_cols:
            del df[col]

    tmp = {
        f'S{whole_dict[single_key][1]}': 'S1',
        f'V{whole_dict[single_key][1]}': 'V1',
        f'V{whole_dict[single_key][1]}_atm': 'V1_atm',
        f'implvol{whole_dict[single_key][1]}': 'implvol1',
        f'S{whole_dict[single_key][1]}_n': 'S1_n',
        f'V{whole_dict[single_key][1]}_n': 'V1_n',
        f'V{whole_dict[single_key][1]}_atm_n': 'V1_atm_n'
    }

    if future_volume:
        tmp[f'volume{whole_dict[single_key][1]}'] = 'volume1'
    df.rename(columns=tmp, inplace=True)



def make_features(df):
    # Make sure this function is called after copying calls for puts.
    # because delta are different for calls and puts.
    df = df.copy()
    df['tau0_implvol0'] = np.sqrt(df['tau0']) * df['implvol0']
    df['sqrt_tau0'] = np.sqrt(df['tau0'])
    df['1_over_sqrt_tau'] = 1 / np.sqrt(df['tau0'])
    
    # the following three features are used for Hull-White regression.
    df['vega_s'] = df['vega_n'] / (df['S0_n'] * np.sqrt(df['tau0']))
    df['delta_vega_s'] = df['delta_bs'] * df['vega_s']
    df['delta2_vega_s'] = (df['delta_bs']**2) * df['vega_s']
    
    return df

    
def tag_data(
        df, tag=None, period=None, 
        offset=None,
        start_date=None, 
        end_date=None
    ):
    """
    Here, we tag for train, validation and test set.
    tag = 0, 1, 2 for train, validation and test respectively.
    The outcoming period will be from the start date to (end_date - business offset)
    """
    if isinstance(offset, pd.Timedelta):
        if offset <= pd.Timedelta('2 hours'):
            t_end = end_date
    else:
        t_end = end_date - offset
    bl = (df['date'] >= start_date) & (df['date'] <= t_end)
    df.loc[bl, f'period{period}'] = tag


def rolling_window(
        df, 
        date_begin=None,
        span_train=None,
        span_val=None,
        span_test=None,
        date_window=None,
        date_end=None,
        offset=None
):
    # These four list are for the benefit of reading results.
    train_date_list = []
    val_date_list = []
    test_date_list = []
    span_end_list = []
    df['Is_In_Some_Test'] = False
    i = 0
    single_period_len = span_train + span_val + span_test
    while (date_begin + single_period_len) < date_end:
        val_date_begin = (date_begin + span_train)
        test_date_begin = (val_date_begin + span_val)
        test_date_end = (test_date_begin + span_test)

        train_date_list.append(date_begin)
        val_date_list.append(val_date_begin)
        test_date_list.append(test_date_begin)
        span_end_list.append(test_date_end)

        df['period{}'.format(i)] = -1
        tag_data(
            df, tag=0, period=i, 
            offset=offset, 
            start_date=date_begin,
            end_date=val_date_begin)
        tag_data(
            df, tag=1, period=i, 
            offset=offset, 
            start_date=val_date_begin,
            end_date=test_date_begin)
        tag_data(
            df, tag=2, period=i, 
            offset=pd.Timedelta(0), 
            start_date=test_date_begin,
            end_date=test_date_end)

        bl_test = df[f'period{i}'] == 2
        df.loc[bl_test, 'Is_In_Some_Test'] = True

        # Update time date for next window.
        date_begin = (date_begin + date_window)
        i += 1

    df_dates = pd.DataFrame({
        'train_date_begin': train_date_list,
        'val_date_begin': val_date_list,
        'test_date_begin': test_date_list,
        'period_end': span_end_list
    })

    return df, df_dates