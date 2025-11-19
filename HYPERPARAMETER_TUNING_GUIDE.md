# Hyperparameter Tuning Guide for Federated Learning with Differential Privacy

## Quick Reference

### Current Settings (From Experiment)
```bash
# DP Settings
--noise_multiplier 1.0      # Too high - causes poor performance
--local_epochs 1            # Too low - insufficient training
--max_grad_norm 1.0         # Reasonable

# ADP Settings
--base_noise 1.0            # Too high initially
--alpha 0.7                 # Decay rate - reasonable
--min_noise 0.05            # Floor value - reasonable
--local_epochs 1            # Too low
```

---

## Understanding the Parameters

### 1. Noise Multiplier (DP) / Base Noise (ADP)

**What it does:** Controls the amount of Gaussian noise added to gradients

**Impact:**
- ↑ Higher → Better privacy (lower epsilon) but worse accuracy
- ↓ Lower → Worse privacy (higher epsilon) but better accuracy

**Recommended Range:**
- Strong Privacy: 0.8 - 1.5 (ε ≈ 1-3)
- Moderate Privacy: 0.4 - 0.8 (ε ≈ 3-8)
- Weak Privacy: 0.1 - 0.4 (ε ≈ 8-20)

**Current Issue:** 1.0 is causing 2x loss increase
**Suggested Fix:** Start with 0.3-0.5

### 2. Alpha (ADP only)

**What it does:** Controls how fast noise decreases over rounds

**Formula:** `noise(round) = max(min_noise, base_noise * exp(-alpha * round))`

**Impact:**
- ↑ Higher α → Faster noise decay, less privacy in later rounds
- ↓ Lower α → Slower decay, more consistent privacy

**Recommended Range:** 0.3 - 0.9

**Example Decay:**
```
alpha = 0.7, base_noise = 1.0, min_noise = 0.05
Round 0: 1.00
Round 1: 0.50
Round 2: 0.25
Round 3: 0.12
Round 4+: 0.05 (min_noise floor)
```

### 3. Local Epochs

**What it does:** Number of passes through local data per round

**Impact:**
- ↑ More epochs → Better convergence but more privacy cost per round
- ↓ Fewer epochs → Faster but may not converge

**Recommended:**
- Without DP: 1-2 epochs
- With DP: 3-5 epochs (noise makes each epoch less effective)

**Current Issue:** 1 epoch insufficient with high noise

### 4. Max Grad Norm (Gradient Clipping)

**What it does:** Clips gradients to this L2 norm before adding noise

**Impact:**
- ↑ Higher → Less clipping, more signal but worse privacy guarantees
- ↓ Lower → More clipping, less signal but better DP guarantees

**Recommended Range:** 0.5 - 2.0
**Current:** 1.0 (reasonable)

### 5. Learning Rate

**What it does:** Step size for optimizer

**Impact with DP:**
- Noise effectively reduces learning rate
- May need to increase LR when using DP

**Recommended:**
- Without DP: 1e-3
- With DP: 1e-2 to 5e-2

### 6. Batch Size

**What it does:** Number of samples per gradient update

**Impact on DP:**
- Larger batches → Better signal-to-noise ratio
- Smaller batches → More noise per sample

**Current:** 32
**Recommended with DP:** 64-128

---

## Tuning Strategies

### Strategy 1: Privacy First (Maximize Privacy)

**Goal:** Achieve strongest privacy with acceptable utility

```bash
# Start conservative
python orchestrator.py --client_script client_dp.py \
  --clients_dir clients --rounds 10 --label readmit_30 \
  --client_args --log-dir logs/dp_privacy_first \
  --noise_multiplier 1.5 --local_epochs 5 --max_grad_norm 0.5

# Check results, if accuracy < 70%, relax slightly
python orchestrator.py --client_script client_dp.py \
  --clients_dir clients --rounds 10 --label readmit_30 \
  --client_args --log-dir logs/dp_privacy_first_v2 \
  --noise_multiplier 1.0 --local_epochs 5 --max_grad_norm 0.8
```

**Target:** ε < 5, Accuracy > 75%

### Strategy 2: Utility First (Maximize Accuracy)

**Goal:** Best accuracy while maintaining some privacy

```bash
# Start aggressive
python orchestrator.py --client_script client_dp.py \
  --clients_dir clients --rounds 10 --label readmit_30 \
  --client_args --log-dir logs/dp_utility_first \
  --noise_multiplier 0.3 --local_epochs 3 --max_grad_norm 1.5

# If privacy budget too high (ε > 20), add more noise
python orchestrator.py --client_script client_dp.py \
  --clients_dir clients --rounds 10 --label readmit_30 \
  --client_args --log-dir logs/dp_utility_first_v2 \
  --noise_multiplier 0.5 --local_epochs 3 --max_grad_norm 1.2
```

**Target:** Accuracy > 85%, ε < 15

### Strategy 3: Balanced (Recommended)

**Goal:** Good tradeoff between privacy and utility

```bash
# DP Balanced
python orchestrator.py --client_script client_dp.py \
  --clients_dir clients --rounds 10 --label readmit_30 \
  --client_args --log-dir logs/dp_balanced \
  --noise_multiplier 0.5 --local_epochs 3 --max_grad_norm 1.0

# ADP Balanced
python orchestrator.py --client_script client_adp.py \
  --clients_dir clients --rounds 10 --label readmit_30 \
  --client_args --log-dir logs/adp_balanced \
  --base_noise 0.6 --alpha 0.5 --min_noise 0.05 --local_epochs 3
```

**Target:** ε ≈ 8-12, Accuracy ≈ 80-85%

---

## Systematic Tuning Process

### Step 1: Establish Baseline
```bash
python orchestrator.py --client_script client_base.py \
  --clients_dir clients --rounds 10 --label readmit_30 \
  --client_args --log-dir logs/baseline_extended
```
**Record:** Baseline accuracy and loss

### Step 2: Test Noise Levels (DP)
```bash
for noise in 0.3 0.5 0.7 1.0 1.5; do
  python orchestrator.py --client_script client_dp.py \
    --clients_dir clients --rounds 5 --label readmit_30 \
    --client_args --log-dir logs/dp_noise_$noise \
    --noise_multiplier $noise --local_epochs 3
done
```

### Step 3: Test Local Epochs
```bash
# Using best noise from step 2
for epochs in 1 3 5; do
  python orchestrator.py --client_script client_dp.py \
    --clients_dir clients --rounds 5 --label readmit_30 \
    --client_args --log-dir logs/dp_epochs_$epochs \
    --noise_multiplier 0.5 --local_epochs $epochs
done
```

### Step 4: Fine-tune Max Grad Norm
```bash
# Using best noise and epochs from above
for norm in 0.5 1.0 1.5 2.0; do
  python orchestrator.py --client_script client_dp.py \
    --clients_dir clients --rounds 5 --label readmit_30 \
    --client_args --log-dir logs/dp_norm_$norm \
    --noise_multiplier 0.5 --local_epochs 3 --max_grad_norm $norm
done
```

### Step 5: Test ADP Alpha Values
```bash
for alpha in 0.3 0.5 0.7 0.9; do
  python orchestrator.py --client_script client_adp.py \
    --clients_dir clients --rounds 5 --label readmit_30 \
    --client_args --log-dir logs/adp_alpha_$alpha \
    --base_noise 0.5 --alpha $alpha --min_noise 0.05 --local_epochs 3
done
```

---

## Interpreting Results

### Good Results Indicators:
- ✅ DP accuracy within 5-10% of baseline
- ✅ ADP accuracy better than DP
- ✅ Loss converging (decreasing over rounds)
- ✅ Epsilon in acceptable range for your privacy needs

### Bad Results Indicators:
- ❌ Accuracy drops > 20% from baseline
- ❌ Loss increasing or flat across rounds
- ❌ Epsilon > 50 (essentially no privacy)
- ❌ Training crashes or NaN values

### Example Analysis:
```
Baseline:  Loss=0.32, Acc=88%, ε=∞
DP (1.0):  Loss=0.64, Acc=73%, ε=7    ← Too much noise
DP (0.5):  Loss=0.38, Acc=84%, ε=12   ← Good balance
DP (0.3):  Loss=0.34, Acc=87%, ε=18   ← Weak privacy
ADP(0.5):  Loss=0.35, Acc=86%, ε=9    ← Best overall
```

---

## Quick Fixes for Common Issues

### Issue: Loss Not Decreasing
**Try:**
- Increase local_epochs to 5
- Reduce noise_multiplier by 0.2
- Increase learning rate to 5e-3

### Issue: Privacy Budget Too High
**Try:**
- Increase noise_multiplier by 0.3
- Reduce max_grad_norm by 0.3
- Reduce number of rounds

### Issue: Training Too Slow
**Try:**
- Reduce local_epochs
- Increase batch size
- Use fewer rounds

### Issue: Model Not Learning
**Try:**
- Start with noise_multiplier = 0.1
- Use 5 local epochs
- Check data preprocessing

---

## Recommended Starting Points

### For Healthcare Data (Your Use Case):
```bash
# Conservative (Strong Privacy)
--noise_multiplier 0.8 --local_epochs 4 --max_grad_norm 0.8

# Moderate (Balanced)
--noise_multiplier 0.5 --local_epochs 3 --max_grad_norm 1.0

# Aggressive (Better Utility)
--noise_multiplier 0.3 --local_epochs 3 --max_grad_norm 1.2
```

### For Different Privacy Requirements:

**HIPAA-like requirements (ε < 10):**
```bash
--noise_multiplier 0.6 --local_epochs 4 --max_grad_norm 0.9
--alpha 0.4  # for ADP
```

**Research setting (ε < 20):**
```bash
--noise_multiplier 0.4 --local_epochs 3 --max_grad_norm 1.0
--alpha 0.6  # for ADP
```

**Proof of concept (ε < 50):**
```bash
--noise_multiplier 0.2 --local_epochs 2 --max_grad_norm 1.5
--alpha 0.8  # for ADP
```

---

## Automation Script

Save this as `tune_hyperparams.sh`:

```bash
#!/bin/bash

NOISE_VALUES="0.3 0.5 0.7"
EPOCH_VALUES="2 3 4"

for noise in $NOISE_VALUES; do
  for epochs in $EPOCH_VALUES; do
    echo "Testing noise=$noise, epochs=$epochs"
    python orchestrator.py --client_script client_dp.py \
      --clients_dir clients --rounds 5 --label readmit_30 \
      --client_args --log-dir logs/tune_n${noise}_e${epochs} \
      --noise_multiplier $noise --local_epochs $epochs
  done
done

# Analyze results
python -c "
import pandas as pd
import glob

results = []
for f in glob.glob('logs/tune_*/client_*.csv'):
    df = pd.read_csv(f)
    last_row = df.iloc[-1]
    results.append({
        'config': f.split('/')[-2],
        'val_acc': last_row['val_acc'],
        'epsilon': last_row['epsilon']
    })

df = pd.DataFrame(results)
df = df.groupby('config').mean()
print(df.sort_values('val_acc', ascending=False))
"
```

Run with: `bash tune_hyperparams.sh`

---

## Key Takeaway

**Start with moderate settings, measure, then adjust:**
1. noise_multiplier = 0.5
2. local_epochs = 3
3. max_grad_norm = 1.0

If accuracy too low → reduce noise
If privacy too weak → increase noise
If not converging → increase epochs
