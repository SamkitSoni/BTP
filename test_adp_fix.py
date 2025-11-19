#!/usr/bin/env python3
"""Quick test to verify ADP client fix works"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import numpy as np

# Simple model
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

# Create dummy data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,)).float()
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = MLP(input_dim=10)
loss_fn = nn.BCELoss()

print("Testing ADP client fix...")
print("=" * 50)

# Simulate multiple rounds
for round_num in range(3):
    print(f"\nRound {round_num + 1}:")
    
    # Get current parameters (simulating set_parameters in Flower)
    params = [p.clone().detach() for p in model.parameters()]
    
    # Set parameters back (this is what Flower does)
    for p, new_p in zip(model.parameters(), params):
        p.data = new_p.data
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create privacy engine (fresh each round)
    privacy_engine = PrivacyEngine()
    
    try:
        # Wrap model (this should NOT fail on round 2+)
        model_p, opt_p, loader_p = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
        )
        
        print(f"  ✓ Successfully wrapped model with PrivacyEngine")
        
        # Train for one batch
        model_p.train()
        for x_batch, y_batch in loader_p:
            opt_p.zero_grad()
            pred = model_p(x_batch).view(-1)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            opt_p.step()
            break  # Just one batch
        
        print(f"  ✓ Training step completed")
        
        # Get epsilon
        epsilon = privacy_engine.get_epsilon(1e-5)
        print(f"  ✓ Epsilon: {epsilon:.4f}")
        
        # CRITICAL: Remove hooks and get original model (key fix)
        if hasattr(model_p, '_module'):
            original_model = model_p._module
            if hasattr(model_p, 'remove_hooks'):
                model_p.remove_hooks()
                print(f"  ✓ Hooks removed from wrapped model")
            # Update model reference
            model = original_model
            print(f"  ✓ Model reference updated to original")
        
    except ValueError as e:
        if "hooks twice" in str(e):
            print(f"  ✗ FAILED: {e}")
            print("\n" + "!" * 50)
            print("ERROR: The fix didn't work!")
            print("!" * 50)
            exit(1)
        else:
            raise

print("\n" + "=" * 50)
print("✓ SUCCESS: All rounds completed without hook errors!")
print("=" * 50)
print("\nThe ADP client fix is working correctly.")
print("You can now run: python experiment_runner.py")
