import scipy
import numpy as np
import pandas as pd

from scipy.integrate import quad
from . import common as cm


def simulate_heston(params):
    mu = params['mu']
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    s0 = params['s0']
    v0 = params['v0']
    time_grid = pd.date_range(
        start=params['start_date'],
        end=params['end_date'],
        freq='B'
    ).to_pydatetime()

    m = len(time_grid)
    path = np.zeros((2, m))
    path[0, 0] = s0
    path[1, 0] = v0

    w0 = np.random.standard_normal(m-1)
    w1 = np.random.standard_normal(m-1)
    w1 = rho * w0 + np.sqrt(1 - rho * 2) * w1

    day_count = 253
    dt = 1 / day_count

    for t in range(1, m):
        """ Euler discretization scheme for S """
        path[0, t] = path[0, t - 1] * (1 + mu * dt +
                                        np.sqrt(path[1, t-1] * dt) * w0[t-1])
        """ Milstein discretization scheme for variance """
        path[1, t] = (path[1, t - 1] + kappa * (theta - path[1, t-1]) * dt +
                        sigma * np.sqrt(path[1, t-1] * dt) * w1[t-1] +
                        1/4. * sigma**2 * dt * (w1[t-1]**2 - 1))

    return pd.DataFrame(
                np.transpose(path),
                index=time_grid,
                columns=['S0', 'Var0']
            )



class ComputeHeston:
    """
    Compute Heston prices by the closed-form formula in Albrecher and Gatheral..
    """
    def __init__(self, kappa, theta, sigma, rho, r):
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.r = r

    def characteristic_func(self, xi, s0, v0, tau):
        ixi = 1j * xi
        d = scipy.sqrt((self.kappa - ixi * self.rho * self.sigma)**2
                       + self.sigma**2 * (ixi + xi**2))
        g = (self.kappa - ixi * self.rho * self.sigma - d) / (self.kappa - ixi * self.rho * self.sigma + d)
        ee = scipy.exp(-d * tau)
        C = ixi * self.r * tau + self.kappa * self.theta / self.sigma**2 * (
            (self.kappa - ixi * self.rho * self.sigma - d) * tau - 2. * scipy.log((1 - g * ee) / (1 - g))
        )
        D = (self.kappa - ixi * self.rho * self.sigma - d) / self.sigma**2 * (
            (1 - ee) / (1 - g * ee)
        )
        return scipy.exp(C + D*v0 + ixi * scipy.log(s0))

    def integ_func(self, xi, s0, v0, K, tau, num):
        ixi = 1j * xi
        if num == 1:
            return (self.characteristic_func(xi - 1j, s0, v0, tau) / (ixi * self.characteristic_func(-1j, s0, v0, tau)) * scipy.exp(-ixi * scipy.log(K))).real
        else:
            return (self.characteristic_func(xi, s0, v0, tau) / (ixi) * scipy.exp(-ixi * scipy.log(K))).real

    def call_price(self, s0, v0, K, tau):
        a = s0 * (0.5 + 1 / scipy.pi * quad(lambda xi: self.integ_func(xi, s0, v0, K, tau, 1), 0., 500.)[0])
        b = K * scipy.exp(-self.r * tau) * (0.5 + 1 / scipy.pi * quad(lambda xi: self.integ_func(xi, s0, v0, K, tau, 2), 0., 500.)[0])
        return a - b


def hs_price_wrapper(
        df, pricer):
    """
    Note: Integration warning comes up if the option is
    deep-in or out-of-money when time-to-maturity is small (although non-zero)
    """
    for ix, row in df.iterrows():
        s, var, tau = row['S0'], row['Var0'], row['tau0']
        k = row['K']
        if tau < 0.0039:
            df.loc[ix, 'V0'] = np.maximum(s - k, 0)
        else:
            price = pricer.call_price(s, var, k, tau)
            df.loc[ix, 'V0'] = price

    return df



