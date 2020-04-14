import copy

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay, DateOffset
from .cleaner_aux import get_time2maturity


class EuropeanOption():
    
    def __init__(
            self,
            path,
            strike,
            op_start_date,
            maturity):
        """
        Star_date is the first date of the option.
        end_date is the end of the simulated path.
        """
        self.strike = strike
        self.op_start_date = op_start_date
        self.maturity = maturity
        self.op_end_date = min(maturity, path.index[-1])
        
        time_grid = pd.date_range(
            start=op_start_date,
            end=self.op_end_date,
            freq='B')
        self.underlying_price = path.loc[time_grid]

        

def find_option_seq_jan_cycle(
        path,
        step_K=5,
        threshold=1,
        strike_per_maturity=None):
    """
    For any individual path, we should create, at any time, the option written on this realisation.
    We follow the Chicago Board Options Exchange rules.
    Parameters:
    ====================================
    underlying: a particular type of simulation object, be it GeometricBrownianMotion or Heston.
    start_date: Timestamp, the time when options start_date to be created.
        we can choose to start_date generate options at a later date than first stock simulation date
    step_K: step of strike prices
    refresh: whether to create a new underlying path
    """

    def get_strike(price):
        """
        Given a stock price, this function determines the endpoints of strike price interval.
        There may need more than 2 strike prices, if the stock price is within the threshold.
        """
        strikes_loc = np.array([np.floor(price / step_K), np.ceil(price / step_K)]) * step_K
        if price - strikes_loc[0] <= threshold:
            strikes_loc = np.insert(strikes_loc, 0, strikes_loc[0] - step_K)

        if strikes_loc[-1] - price <= threshold:
            strikes_loc = np.append(strikes_loc, strikes_loc[1] + step_K)

        return np.unique(strikes_loc)


    def generate_option(
            path,
            start_date,
            strike,
            maturity):
        
        name = 'C' + str(start_date.date()) + '-' + str(maturity.date()) + '-' + str(strike)
        options[name] = EuropeanOption(
            path=path,
            strike=strike,
            op_start_date=start_date,
            maturity=maturity
        )


    def get_new_strike(price_now, strike_min_max):
        """
        This function is different to get_strike.
        When the spot price jump out of the range (K_min, K_max) by a large number,
        let's say the increase is more than 2 * step_k, from 281 to 291.
        Adding a step_k to K_max (285 currently) only give 290, which is not enough to cover.
        """
        K_min = strike_min_max[0]
        K_max = strike_min_max[-1]
        if price_now > K_max + 0.0001:
            a = np.arange(K_max, price_now + step_K + 0.0001, step_K).tolist()
            a.remove(K_max)

        if price_now < K_min - 0.0001:
            # K_min is not in the a, right end not included.
            a = np.arange(K_min, price_now - step_K - 0.0001, -step_K).tolist()
            a.remove(K_min)
        return a


    start_date = path.index[0] 
    # this offset allows us to generate an option with maturity longer than underlying final date.
    end = path.index[-1] + DateOffset(years=1)

    # the 3rd Friday every month is expiries.
    expiry = pd.date_range(start=start_date, end=end, freq='WOM-3FRI')
        
    if strike_per_maturity is None:
        strike_per_maturity = {}
        cur = path.loc[start_date, 'S0']
        strike_bracket = get_strike(cur)
        for T in expiry[:12]:
            strike_per_maturity[T] = [strike_bracket[0], strike_bracket[-1]]
    else:
        strike_per_maturity = copy.deepcopy(strike_per_maturity)
    
    options = {}
    for mat, value in strike_per_maturity.items():
        strike_bracket = np.arange(value[0], value[1]+0.0001, step_K)
        for k in strike_bracket:        
            generate_option(path, start_date, k, mat)
    

    # we iterate over each day, and act according to the rules.
    time_grid = path.index

    for t in time_grid:
        price_now = path.loc[t, 'S0']

        # judge whether the stock price will move out of strike bracket bounds, for each maturity.
        for mat, value in strike_per_maturity.items():
            if mat > t + BDay():
                if price_now < value[0] + 0.0001:
                    new_strikes = get_new_strike(price_now, strike_min_max=value)
                    strike_per_maturity[mat][0] = min(new_strikes)
                    for k in new_strikes:
                        generate_option(path, t + BDay(), k, mat)

                elif price_now > value[1] - 0.0001:
                    new_strikes = get_new_strike(price_now, strike_min_max=value)
                    strike_per_maturity[mat][1] = max(new_strikes)
                    for k in new_strikes:
                        generate_option(path, t + BDay(), k, mat)


        # check for each time, to see if any option expires.
        # option expires IFF it is an expiration date.
        if t in expiry:
            expiry = expiry[1:]
            del strike_per_maturity[t]
            new_maturity = expiry[11]

            # then we need to get the strikes for generating options.
            strike_bracket = get_strike(price_now)
            strike_per_maturity[new_maturity] = [strike_bracket[0], strike_bracket[-1]]

            # then we generate new options with these strikes and new maturity.
            for k in strike_bracket:
                generate_option(path, t + BDay(), k, new_maturity)

    return options, strike_per_maturity



def get_hedge_df(
        options,
        interest_rate,
        display=True):
    """
    this methods returns a dataframe whose columns are
    current stock price, strike price, time-to-maturity, etc.
    Parameters:
    ============================
    options:
        a dictionary of all options during a time interval
    ============================
    """
    df = pd.DataFrame()
    optionID_counter = 1
    
    for name, op in options.items():
        if op.op_start_date > op.op_end_date:
            continue  # exit this iter and do next iter
        # option should have at least one day left to maturity.
        df_add = op.underlying_price.copy()

        df_add.loc[:, 'K'] = op.strike
        
        df_add.loc[:, 'tau0'] = get_time2maturity(
                df_add.index,
                [op.maturity]
            )
        df_add['optionid'] = optionID_counter
        optionID_counter += 1
            
        df = pd.concat([df, df_add])
        if display:
            print(name + ': Done')

    df['short_rate'] = interest_rate
    df['M0'] = df['S0'] / df['K']
    df = df.reset_index()
    df.rename(columns={'index': 'date'}, inplace=True)
    return df

