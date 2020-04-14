import pandas as pd
import numpy as np

from . import common as cm

def simulate_geometric_bm(params):
    s0 = params['s0']
    vol = params['volatility']
    log_ret = params['mu'] - 0.5 * vol**2
    time_grid = pd.date_range(
        start=params['start_date'],
        end=params['end_date'],
        freq='B'
    ).to_pydatetime()
    
    M = len(time_grid)
    paths = np.zeros(M)
    paths[0] = s0
    
    rand = np.random.standard_normal(M - 1)
    day_count = 253
    dt = 1 / day_count

    for t in range(1, M):
        ran = rand[t - 1]
        paths[t] = paths[t - 1] * np.exp(
            log_ret * dt + vol * np.sqrt(dt) * ran)
    return pd.DataFrame(paths, index=time_grid, columns=['S0'])

