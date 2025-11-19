#!/usr/bin/env python3
"""Check if round logging is working properly"""

import pandas as pd
import glob
import os

print("Checking round logging in experiment results...\n")
print("="*60)

for exp in ["baseline", "dp", "adp"]:
    log_dir = f"logs/{exp}"
    if not os.path.exists(log_dir):
        print(f"\n{exp.upper()}: Log directory not found yet")
        continue
    
    csv_files = glob.glob(f"{log_dir}/client_*.csv")
    if not csv_files:
        print(f"\n{exp.upper()}: No log files yet")
        continue
    
    print(f"\n{exp.upper()}:")
    print("-"*60)
    
    for csv_file in sorted(csv_files)[:2]:  # Check first 2 clients
        try:
            df = pd.read_csv(csv_file)
            client_name = os.path.basename(csv_file)
            
            if 'round' in df.columns:
                rounds = df['round'].tolist()
                print(f"  {client_name}: rounds = {rounds}")
                
                if all(r == -1 or r == 0 for r in rounds):
                    print(f"    ⚠️  WARNING: Round numbers not working!")
                elif rounds == list(range(1, len(rounds)+1)):
                    print(f"    ✓ Round numbers working correctly!")
                else:
                    print(f"    ⚠️  Round numbers look unusual")
            else:
                print(f"  {client_name}: No 'round' column found")
        except Exception as e:
            print(f"  {client_name}: Error reading - {e}")

print("\n" + "="*60)
print("\nTo view full logs, check:")
print("  - logs/baseline/client_0.csv")
print("  - logs/dp/client_0.csv") 
print("  - logs/adp/client_0.csv")
