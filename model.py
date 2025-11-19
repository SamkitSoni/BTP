# model.py
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=128, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(8, hidden//2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden//2), out_dim)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
