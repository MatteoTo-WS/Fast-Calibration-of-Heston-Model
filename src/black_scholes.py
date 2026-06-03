import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# call price with Black-Scholes
def black_scholes_call(S, K, T, r, sigma):   
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(C_mkt, S, K, T, r):
    if C_mkt <= max(0, S - K * np.exp(-r * T)):
        return np.nan
    # objective function: BS_price(sigma)-market_price=0
    objective_function = lambda sigma: black_scholes_call(S, K, T, r, sigma) - C_mkt
    try:
        return brentq(objective_function, 1e-5, 5.0)
    except ValueError:
        return np.nan