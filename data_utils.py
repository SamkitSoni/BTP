# data_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

def load_local_csv(path, feature_cols, label_col="readmit_30", test_size=0.2, seed=0):
    df = pd.read_csv(path)
    X = df[feature_cols].values.astype(float)
    y = df[label_col].values.astype(float)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y if len(np.unique(y))>1 else None)
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    return train_ds, val_ds

def make_loader(ds, batch_size=32, shuffle=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
