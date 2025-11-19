# client_adp.py - Adaptive Differential Privacy client
# Compatible with: Flower >= 1.10, Opacus >= 1.4, PyTorch >= 2.0

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import pandas as pd
import argparse
import json
from torch.utils.data import DataLoader, TensorDataset
import os
import math


# ---------------------- MODEL ----------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


# ------------------ EVALUATION ---------------------
def evaluate(model, loader):
    loss_fn = nn.BCELoss()
    total_loss = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).view(-1)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            preds = (pred > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

    return total_loss / len(loader), correct / total


# ------------------- ADP CLIENT --------------------
class ADPClient(fl.client.NumPyClient):

    def __init__(self, model, train_loader, val_loader,
                 client_id, log_dir,
                 base_noise, alpha, min_noise,
                 max_grad_norm, epochs):

        self.model = model  # Keep original unwrapped model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.client_id = client_id
        self.base_noise = base_noise
        self.alpha = alpha
        self.min_noise = min_noise
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs

        # Log file
        self.log_path = os.path.join(log_dir, f"client_{client_id}.csv")
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("round,val_loss,val_acc,epsilon,noise\n")

    # Flower API ----------------------
    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, params):
        for p, new in zip(self.model.parameters(), params):
            p.data = torch.tensor(new, dtype=p.dtype)

    # Adaptive noise schedule ----------
    def compute_noise(self, round_num):
        # Use round_num - 1 so first round (round_num=1) starts with base_noise
        # Then decays: round 1->0.6, round 2->0.364, round 3->0.221, etc.
        noise = max(self.min_noise,
                    self.base_noise * math.exp(-self.alpha * (round_num - 1)))
        return noise

    # Federated local training -----------
    def fit(self, params, config):
        # Set parameters on the original model
        self.set_parameters(params)

        # Fix: Get round from config, try multiple keys
        round_num = config.get("server_round", config.get("round", 0))
        noise_multiplier = self.compute_noise(round_num)

        # Reset optimizer every round
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        # ---------------------------
        # OPACUS: Create fresh privacy engine each round
        # ---------------------------
        self.model.train()
        
        # Create a fresh privacy engine each round
        privacy_engine = PrivacyEngine()

        # Wrap model, optimizer, and dataloader with privacy
        model_p, opt_p, loader_p = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        # Local training epochs
        for _ in range(self.epochs):
            model_p.train()
            for x, y in loader_p:
                opt_p.zero_grad()
                pred = model_p(x).view(-1)
                loss = loss_fn(pred, y)
                loss.backward()
                opt_p.step()

        # Get epsilon before removing hooks
        epsilon = privacy_engine.get_epsilon(1e-5)

        # CRITICAL: Remove Opacus hooks from the model to prevent "hooks twice" error
        if hasattr(model_p, '_module'):
            # Access the original unwrapped model
            original_model = model_p._module
            # Remove hooks from the wrapped model
            if hasattr(model_p, 'remove_hooks'):
                model_p.remove_hooks()
            # Update self.model to point to the clean original
            self.model = original_model
        
        # Validation on clean model
        val_loss, val_acc = evaluate(self.model, self.val_loader)

        # Logging
        with open(self.log_path, "a") as f:
            f.write(f"{round_num},{val_loss},{val_acc},{epsilon},{noise_multiplier}\n")

        return self.get_parameters(), len(self.train_loader.dataset), {}

    # Evaluation phase ---------------
    def evaluate(self, params, config):
        self.set_parameters(params)
        val_loss, val_acc = evaluate(self.model, self.val_loader)
        return float(val_loss), len(self.val_loader.dataset), {
            "val_acc": val_acc
        }


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--label", default="readmit_30")

    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--log-dir", default="logs")

    # ADP hyperparameters
    parser.add_argument("--base_noise", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--min_noise", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--local_epochs", type=int, default=1)

    args = parser.parse_args()

    # Auto-load features if not provided
    if args.features is None or len(args.features) == 0:
        with open("data/features.json", "r") as f:
            args.features = json.load(f)
        print(f"[AUTO] Loaded {len(args.features)} features.")

    # Load data
    df = pd.read_csv(args.data)
     # 1) Ensure all expected features exist
    missing = [f for f in args.features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in client data: {missing}")

    # 2) Try to coerce each column to numeric; collect any failures
    bad_cols = []
    for col in args.features:
        # coerce, keep the coerced series but also detect if many values became NaN
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_nan = coerced.isna().sum()
        # if everything becomes NaN or many become NaN, mark it
        if n_nan == len(coerced):
            bad_cols.append((col, "all_non_numeric"))
        df[col] = coerced

    if bad_cols:
        # give a helpful error so you can inspect the client CSV immediately
        msg = "Found non-numeric feature columns in client data:\n"
        msg += "\n".join([f"- {c}: {reason}" for c,reason in bad_cols])
        msg += "\n(Inspect the client CSV or run the diagnostic script.)"
        raise ValueError(msg)

    # 3) fill remaining NaNs with column medians (or 0 if median NaN)
    for col in args.features:
        median = pd.Series(df[col]).median()
        if pd.isna(median):
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(median)

    # 4) final conversion to numpy float32 then tensor
    X_np = df[args.features].to_numpy(dtype=np.float32, copy=True)
    X = torch.from_numpy(X_np)
    # --- end safe conversion ---
    y = torch.tensor(df[args.label].values, dtype=torch.float32)

    split = int(0.8 * len(df))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Init model
    model = MLP(input_dim=X.shape[1])

    # Create ADP client
    client = ADPClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        client_id=args.client_id,
        log_dir=args.log_dir,
        base_noise=args.base_noise,
        alpha=args.alpha,
        min_noise=args.min_noise,
        max_grad_norm=args.max_grad_norm,
        epochs=args.local_epochs,
    )

    # Start client (NEW Flower API)
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )
