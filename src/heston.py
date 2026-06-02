import numpy as np
from scipy.stats import norm
import scipy.integrate as integrate

def heston_simulations(S_0, V_0, kappa, theta, xi, rho, r, T, N, n_paths):
    # Simulazione numerica dell andamento del processo di Heston.
    dt = T/N

    S = np.zeros((N + 1, n_paths))
    V = np.zeros((N + 1, n_paths))

    S[0, :] = S_0
    V[0, :] = V_0

    for t in range(1, N + 1):
        Z_1 = np.random.normal(0, 1, n_paths)
        Z_2 = np.random.normal(0, 1, n_paths)
        
        dW_S = np.sqrt(dt) * Z_1
        dW_V = np.sqrt(dt) * (rho * Z_1 + np.sqrt(1 - rho**2) * Z_2)

        V_plus = np.maximum(V[t-1, :], 0) 

        S[t, :] = S[t-1, :] + r * S[t-1, :] * dt + np.sqrt(V_plus) * S[t-1, :] * dW_S
        V[t, :] = V[t-1, :] + kappa * (theta - V_plus) * dt + xi * np.sqrt(V_plus) * dW_V
    # crazy, mi sembra di scrivere su Matlab, che ricordi
    return S, V 

def heston_characteristic_function(u, S0, V0, kappa, theta, xi, rho, r, T, j):
    # calcolo funzione caratteristica necessaria per il calcolo dell integrale e quidni del prezzo
    x = np.log(S0)
    a = kappa * theta
  
    if j == 1:
        b = kappa - rho * xi
        u_i = 1j * u
    else:
        b = kappa
        u_i = -1j * u
        
    d = np.sqrt((rho * xi * u * 1j - b)**2 - xi**2 * (2 * u_i - u**2))
    g = (b - rho * xi * u * 1j + d) / (b - rho * xi * u * 1j - d)
    
    C = r * u * 1j * T + (a / xi**2) * ((b - rho * xi * u * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = ((b - rho * xi * u * 1j + d) / xi**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    
    return np.exp(C + D * V0 + 1j * u * x)

def heston_probability_integrand(u, S0, V0, kappa, theta, xi, rho, r, T, K, j):
    cf = heston_characteristic_function(u, S0, V0, kappa, theta, xi, rho, r, T, j)
    # Invertiamo
    return np.real(np.exp(-1j * u * np.log(K)) * cf / (1j * u))

def heston_call_price(S0, K, T, r, kappa, theta, xi, rho, V0):
    int_1 = lambda u: heston_probability_integrand(u, S0, V0, kappa, theta, xi, rho, r, T, K, j=1)
    int_2 = lambda u: heston_probability_integrand(u, S0, V0, kappa, theta, xi, rho, r, T, K, j=2)
    
    # Integrazione numerica tra (0, 100]
    P1 = 0.5 + (1 / np.pi) * integrate.quad(int_1, 1e-5, 100, limit=200)[0]
    P2 = 0.5 + (1 / np.pi) * integrate.quad(int_2, 1e-5, 100, limit=200)[0]
    
    # Formula di pricing di Heston in forma chiusa derivante dal Thm di inversione di Gil-Pealez 
    price = S0 * P1 - K * np.exp(-r * T) * P2
    
    # Arbitrage constraint: un'opzione non può avere prezzo negativo
    return max(0.0, price)