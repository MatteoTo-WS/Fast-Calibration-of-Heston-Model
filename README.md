## Fast-Calibration-of-Heston-Model

This repository contains a framework designed to calibrate the **Heston Stochastic Volatility Model** to real S&P 500 (`^SPX`) market options data. Instead of relying on slow traditional numerical integration we implement a NN model to approximate the IV instantaneously. This enables a fast real-time calibration with the SciPy optimizer.

Steps of the calibration:
1. **Synthetic data**: build a large dataset of artificial options by calculating their theoretical Black-Scholes IV using different tuples of random Heston parameters;
2. **NN pricer**: train a NN to learn the direct mapping from the 5 Heston parameters + strike + maturity to the target IV and use it as a fast option pricer;
3. **Calibration**: pass real S&P 500 options data into a SciPy optimizer to find the 5 optimal Heston parameters that best fit the actual market volatility surface.

## Structure

```text
├── data/
│   ├── heston_synthetic_dataset.csv  # Synthetic dataset for neural network training
│   ├── SPX_data_26.02.24.csv         # Original data
│   └── SPX_data_clean_IV.csv         # Cleaned real S&P 500 market options data
├── models/
│   ├── heston_pytorch_model.pth      # Weights of the trained NN
│   └── heston_pytorch_scaler.pkl     # Fitted StandardScaler
├── src/
│   ├── black_scholes.py              # BS call and IV computation
│   ├── heston.py                     # Heston characteristic function
│   └── nn_for_volatility.py          # NN architecture
├── heston_calibration.ipynb          # main notebook
└── README.md                 