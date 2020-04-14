import pandas as pd
import numpy as np

def simulate_fake_vix(VIXPARAS, start_date, end_date):
    """
    Simulates OU process following 
    dx_t = kappa * (mu - x_t) dt + sigma * dW_t
    """
    x0 = VIXPARAS['vix0']
    kappa = VIXPARAS['kappa']
    sigma = VIXPARAS['sigma']
    mu = VIXPARAS['mu']
    
    time_grid = pd.date_range(start=start_date, end=end_date, freq='B').to_pydatetime()
    
    M = len(time_grid)
    path = np.zeros(M)
    path[0] = x0
    rand = np.random.standard_normal(M-1)
    day_count = 253
    dt = 1 / day_count
        
    for t in range(1, len(time_grid)):
        path[t] = path[t-1] + kappa * (mu - path[t-1]) * dt \
                + sigma * np.sqrt(dt) * rand[t - 1]
    return pd.DataFrame(path, time_grid, columns=['fake_vix'])             
 
        
        
        
        
        
        
        
        