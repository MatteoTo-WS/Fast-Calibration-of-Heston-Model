import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class heston_volatility_nn(nn.Module):
    def __init__(self):
        super(heston_volatility_nn, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            # output: implied volatility (must be positive)
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        return self.network(x)


def train_heston_model(csv_path='data/heston_synthetic_dataset.csv', epochs=60, batch_size=64, patience=5):
    """
    upload synthetic dataset, normalize and train the nn with early stopping, then save the model
    """
        
    # upload and split data
    df = pd.read_csv(csv_path)

    x_cols = ['kappa', 'theta', 'xi', 'rho', 'V0', 'moneyness', 'time_to_maturity']
    x = df[x_cols].values

    y = (df['target_iv'].values * 100.0).astype(np.float32).reshape(-1, 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # normalize
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # convert to pytorch tensors
    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    model = heston_volatility_nn()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(3, patience // 3))
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_weights = None
    
    num_samples = x_train_t.size(0)
    
    print("Start training.")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        # shuffle the training data
        permutation = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train_t[indices], y_train_t[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= num_samples
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()

        scheduler.step(val_loss)

        # print every 5 epoches 
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
            
        # early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_model_weights = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopped at epoch {epoch+1}.")
                model.load_state_dict(best_model_weights)
                break
                
    torch.save(model.state_dict(), 'models/heston_pytorch_model.pth')
    joblib.dump(scaler, 'models/heston_pytorch_scaler.pkl')
    return model, scaler


def heston_pytorch_objective(params, df_market, model, scaler):
    """
    Compute the objective function for the calibration of the Heston model. 

    inputs:
     - params: list of Heston parameters [kappa, theta, xi, rho, V0]
     - df_market: DataFrame containing market option data
     - model: pre-trained NN
     - scaler: StandardScaler fitted on the synthetic dataset

    output:
     - MSE: between market IV and IV predicted by the NN

    """
    kappa, theta, xi, rho, V0 = params
    # penalty for financial constraints
    if kappa <= 0 or theta <= 0 or xi <= 0 or rho <= -1.0 or rho >= 1.0 or V0 <= 0:
        return 1e10
        
    n_options = len(df_market)
    inputs = np.zeros((n_options, 7))
    inputs[:, 0] = kappa
    inputs[:, 1] = theta
    inputs[:, 2] = xi
    inputs[:, 3] = rho
    inputs[:, 4] = V0
    inputs[:, 5] = (df_market['strike'] / df_market['S_0']).values
    inputs[:, 6] = df_market['time_to_maturity'].values
    # scale and convert to PyTorch tensor
    inputs_scaled = scaler.transform(inputs)
    inputs_t = torch.tensor(inputs_scaled, dtype=torch.float32)
    # inference
    model.eval()
    with torch.no_grad():
        nn_predictions = model(inputs_t).numpy().flatten()    
    # MSE
    market_iv = df_market['implied_vol'].values * 100 
    mse = np.mean((market_iv - nn_predictions) ** 2)   
    return mse
