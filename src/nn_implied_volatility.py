"""
Rete neurale densa a 4 layer (Softplus) per approssimare la IV del modello di Heston.
Input:  (kappa, theta, xi, rho, V0, K, T)
Output: sigma_IV
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy.optimize import brentq

# ---- Parametri di mercato (coerenti con il dataset sintetico) ----
S0 = 6852.66
r = 0.045


# ---- Black-Scholes per inversione IV ----

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K * np.exp(-r * T))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def compute_iv(price, S, K, T, r):
    intrinsic = max(0.0, S - K * np.exp(-r * T))
    if price <= intrinsic or price >= S:
        return np.nan
    try:
        return brentq(lambda sig: bs_call_price(S, K, T, r, sig) - price, 1e-5, 5.0)
    except ValueError:
        return np.nan


# ---- Definizione della rete ----

class IVNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Softplus(),
            nn.Linear(128, 64),        nn.Softplus(),
            nn.Linear(64, 32),         nn.Softplus(),
            nn.Linear(32, 1),          nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


# ---- Funzione per caricare il modello salvato e predire ----

def load_trained_model(path="models/iv_net.pth"):
    """Carica il modello salvato e restituisce una funzione predict(kappa, theta, xi, rho, V0, K, T)."""
    checkpoint = torch.load(path, weights_only=False)
    model = IVNet(input_dim=7)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # media e dev.std del training set per normalizzare gli input
    mean = checkpoint["scaler_mean"]
    scale = checkpoint["scaler_scale"]

    def predict_iv(kappa, theta, xi, rho, V0, K, T):
        X = np.array([[kappa, theta, xi, rho, V0, K, T]], dtype=np.float32)
        X_scaled = (X - mean) / scale  # stessa normalizzazione del training
        with torch.no_grad():
            iv = model(torch.tensor(X_scaled, dtype=torch.float32))
        return iv.item()

    return predict_iv


# ===========================================================================
#  MAIN: training e salvataggio
# ===========================================================================

if __name__ == "__main__":

    # ---- 1. Caricamento dati sintetici ----
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "heston_synthetic_dataset.csv")
    df = pd.read_csv(csv_path)
    print(f"Dataset sintetico caricato: {len(df)} righe")

    # ---- 2. Calcolo IV (inversione BS) ----
    from tqdm import tqdm
    tqdm.pandas(desc="Calcolo IV")

    df["IV"] = df.progress_apply(
        lambda row: compute_iv(row["target_price"], S0, row["strike"], row["time_to_maturity"], r),
        axis=1,
    )
    df = df.dropna(subset=["IV"])
    print(f"Righe valide dopo calcolo IV: {len(df)}")

    # ---- 3. Preparazione feature e target ----
    feature_cols = ["kappa", "theta", "xi", "rho", "V0", "strike", "time_to_maturity"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["IV"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_sc, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    # ---- 4. Training ----
    model = IVNet(input_dim=7)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

    epochs = 800
    train_losses, test_losses = [], []

    print(f"\nTraining: {len(X_train)} train / {len(X_test)} test, {epochs} epoche")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            running += loss.item() * bx.size(0)

        train_loss = running / len(X_train)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test_t), y_test_t).item()
        test_losses.append(test_loss)
        scheduler.step(test_loss)

        if epoch % 100 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoca {epoch:4d}/{epochs} | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f} | LR: {lr_now:.2e}")

    # ---- 5. Risultati finali ----
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()

    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print(f"\nRisultati sul Test Set:  MSE={test_losses[-1]:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}")

    # ---- 6. Salvataggio modello (pesi + normalizzazione in un unico file) ----
    save_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "iv_net.pth")
    torch.save({
        "model": model.state_dict(),
        "scaler_mean": scaler.mean_,    # media di ogni feature nel training set
        "scaler_scale": scaler.scale_,  # dev.std di ogni feature nel training set
    }, model_path)
    print(f"\nModello salvato in: {model_path}")

    # ---- 7. Grafici ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train MSE")
    axes[0].plot(test_losses, label="Test MSE")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Curva di Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test, y_pred, alpha=0.4, s=10)
    lims = [min(y_test.min(), y_pred.min()) * 0.95, max(y_test.max(), y_pred.max()) * 1.05]
    axes[1].plot(lims, lims, "r--", label="Perfetta corrispondenza")
    axes[1].set_xlabel("IV Heston")
    axes[1].set_ylabel("IV NN")
    axes[1].set_title("IV Heston vs IV NN (Test)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.show()

    # ---- 8. Esempio di utilizzo del modello salvato ----
    print("\n--- Test del modello salvato ---")
    predict = load_trained_model(model_path)

    # Esempio: prediciamo la IV per un set di parametri Heston
    iv_pred = predict(kappa=2.0, theta=0.05, xi=0.3, rho=-0.7, V0=0.04, K=6800.0, T=0.5)
    print(f"  predict(kappa=2.0, theta=0.05, xi=0.3, rho=-0.7, V0=0.04, K=6800, T=0.5)")
    print(f"  -> IV predetta: {iv_pred:.6f}")
 