# client_dp.py (fixed for new Opacus + Flower)

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


# ---------------- Model -----------------
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


# -------- DP Client ----------
class DPFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, client_id, log_dir,
                 noise_multiplier, max_grad_norm, epochs):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id
        self.epochs = epochs

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # NEW OPACUS API
        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        self.log_path = os.path.join(log_dir, f"client_{client_id}.csv")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("round,val_loss,val_acc,epsilon,noise_multiplier\n")

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, params):
        for p, new in zip(self.model.parameters(), params):
            p.data = torch.tensor(new, dtype=p.dtype)

    def fit(self, params, config):
        self.set_parameters(params)

        loss_fn = nn.BCELoss()

        for _ in range(self.epochs):
            self.model.train()
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(x).view(-1)
                loss = loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()

        val_loss, val_acc = evaluate(self.model, self.val_loader)

        epsilon = self.privacy_engine.get_epsilon(1e-5)

        # Fix: Get round from config, try multiple keys
        round_num = config.get("server_round", config.get("round", -1))
        
        with open(self.log_path, "a") as f:
            f.write(f"{round_num},{val_loss},{val_acc},{epsilon},{config.get('noise_multiplier', 'NA')}\n")

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, params, config):
        self.set_parameters(params)
        val_loss, val_acc = evaluate(self.model, self.val_loader)
        return float(val_loss), len(self.val_loader.dataset), {"val_acc": val_acc}



def evaluate(model, loader):
    loss_fn = nn.BCELoss()
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            pred = model(x).view(-1)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            preds = (pred > 0.5).float()
            correct += (preds == y).sum().item()
            total += len(y)

    return total_loss / len(loader), correct / total



# ---------------- MAIN -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--label", default="readmit_30")
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--noise_multiplier", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()

    # Auto-load features
    if args.features is None or len(args.features) == 0:
        with open("data/features.json", "r") as f:
            args.features = json.load(f)
        print(f"[AUTO] Loaded {len(args.features)} features.")

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

    model = MLP(input_dim=X.shape[1])

    client = DPFlowerClient(
        model, train_loader, val_loader,
        client_id=args.client_id,
        log_dir=args.log_dir,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        epochs=args.local_epochs
    )

    # New Flower API
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
    )
