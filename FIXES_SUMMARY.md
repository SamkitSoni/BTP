# Federated Learning Experiment - Issues and Fixes

## Analysis Date
November 17, 2025

## Experiment Results Summary

### 1. BASELINE Experiment ✅ SUCCESS
- **Status**: Completed successfully
- **Rounds**: 5
- **Final Loss**: 0.3176
- **Training Time**: 58.3 seconds
- **Issues**: None

### 2. DP (Differential Privacy) Experiment ⚠️ WARNINGS
- **Status**: Completed with warnings
- **Rounds**: 5
- **Final Loss**: 0.6407 (much higher than baseline)
- **Training Time**: 186.7 seconds (3x slower)
- **Issues**:
  - Full backward hook warnings (non-critical)
  - Very high loss compared to baseline suggests excessive noise
  - Secure RNG turned off (expected for experimentation)

### 3. ADP (Adaptive DP) Experiment ❌ FAILED
- **Status**: Failed at round 3
- **Error**: `ValueError: Trying to add hooks twice to the same model`
- **Root Cause**: Privacy engine hooks being reapplied each round
- **Impact**: Experiment terminated prematurely

---

## Critical Issues

### Issue #1: ADP Client - Double Hook Registration ❌ CRITICAL

**Problem:**
```python
ValueError: Trying to add hooks twice to the same model
```

**Location:** `client_adp.py`, line 110 in `fit()` method

**Root Cause:**
- Privacy engine creates a new `PrivacyEngine()` each round
- `make_private()` adds backward hooks to the model
- On subsequent rounds, hooks already exist → error
- The wrapped model is not properly unwrapped between rounds

**Fix Applied:**
1. Track privacy engine as instance variable
2. Unwrap model before reapplying privacy engine
3. Handle wrapped model in evaluation

**Code Changes:**
```python
# In __init__:
self.privacy_engine = None  # Track privacy engine

# In fit():
# Remove existing hooks if they exist
if self.privacy_engine is not None:
    if hasattr(self.model, '_module'):
        self.model = self.model._module

# Create fresh privacy engine each round
self.privacy_engine = PrivacyEngine()
model_p, opt_p, loader_p = self.privacy_engine.make_private(...)
self.model = model_p  # Update reference

# In evaluate():
eval_model = self.model._module if hasattr(self.model, '_module') else self.model
val_loss, val_acc = evaluate(eval_model, self.val_loader)
```

---

### Issue #2: DP Client - Performance Degradation ⚠️ WARNING

**Problem:**
- DP loss (0.64) is 2x higher than baseline (0.32)
- Training is 3x slower
- Model is not learning effectively

**Possible Causes:**
1. **Excessive noise**: `noise_multiplier=1.0` may be too high
2. **Insufficient training**: Only 1 local epoch per round
3. **Learning rate**: May need adjustment with DP noise
4. **Batch size**: Small batches (32) increase noise impact

**Suggested Fixes:**

#### Option A: Reduce Noise
```python
# In experiment_runner.py, reduce noise multiplier
--noise_multiplier 0.5  # Instead of 1.0
```

#### Option B: Increase Local Training
```python
--local_epochs 3  # Instead of 1
```

#### Option C: Adjust Learning Rate
```python
# In client_dp.py and client_adp.py
optimizer = optim.Adam(self.model.parameters(), lr=1e-2)  # Increase from 1e-3
```

#### Option D: Increase Batch Size
```python
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)  # Instead of 32
```

---

### Issue #3: Backward Hook Warnings ⚠️ NON-CRITICAL

**Problem:**
```
UserWarning: Full backward hook is firing when gradients are computed with respect 
to module outputs since no inputs require gradients.
```

**Location:** Both `client_dp.py` and `client_adp.py`

**Root Cause:**
- Opacus hook system warning
- Input tensors don't require gradients (data tensors)
- This is expected behavior but generates warnings

**Fix (Optional):**
```python
# Suppress specific warning if desired
import warnings
warnings.filterwarnings('ignore', message='Full backward hook is firing')
```

**Note:** This is a cosmetic issue and doesn't affect functionality.

---

## Recommended Action Plan

### Immediate Actions (Fix Blocking Issues)

1. **✅ DONE** - Fixed ADP client hook issue
2. **Test ADP Fix:**
   ```bash
   python experiment_runner.py
   ```

### Performance Tuning (If Still Poor Results)

3. **Tune DP Hyperparameters:**
   ```bash
   # Test with reduced noise
   python orchestrator.py --client_script client_dp.py \
     --clients_dir clients --rounds 5 --label readmit_30 \
     --client_args --log-dir logs/dp_tuned \
     --noise_multiplier 0.3 --local_epochs 3
   ```

4. **Tune ADP Hyperparameters:**
   ```bash
   # Test with adjusted adaptive schedule
   python orchestrator.py --client_script client_adp.py \
     --clients_dir clients --rounds 5 --label readmit_30 \
     --client_args --log-dir logs/adp_tuned \
     --base_noise 0.5 --alpha 0.5 --min_noise 0.05 --local_epochs 3
   ```

5. **Compare Results:**
   - Check `results/summary.csv`
   - Compare final losses and accuracies
   - Evaluate privacy-utility tradeoff

---

## Additional Recommendations

### Code Quality

1. **Add Error Handling:**
```python
try:
    model_p, opt_p, loader_p = self.privacy_engine.make_private(...)
except Exception as e:
    print(f"Error in make_private: {e}")
    # Fallback or cleanup logic
    raise
```

2. **Add Logging:**
```python
import logging
logging.info(f"Round {round_num}: noise={noise_multiplier:.4f}, epsilon={epsilon:.4f}")
```

3. **Model State Validation:**
```python
def validate_model_state(self):
    """Check if model has privacy hooks attached"""
    return hasattr(self.model, '_module')
```

### Privacy Budget Tracking

4. **Monitor Epsilon Growth:**
   - Track cumulative epsilon across rounds
   - Set epsilon budget threshold
   - Stop training if budget exceeded

```python
EPSILON_BUDGET = 10.0
cumulative_epsilon = 0
for round in range(num_rounds):
    # ... training ...
    cumulative_epsilon += epsilon
    if cumulative_epsilon > EPSILON_BUDGET:
        print("Privacy budget exhausted!")
        break
```

### Experiment Design

5. **Add Baseline Comparisons:**
   - Track relative performance drop with DP
   - Expected: 5-15% accuracy drop with reasonable privacy
   - Current: ~50% performance drop suggests over-privatization

6. **Grid Search Hyperparameters:**
```python
noise_values = [0.1, 0.3, 0.5, 1.0]
epochs_values = [1, 3, 5]
for noise in noise_values:
    for epochs in epochs_values:
        run_experiment(noise, epochs)
```

---

## Testing Checklist

- [x] Fix ADP hook issue
- [ ] Run full experiment suite
- [ ] Verify all 3 experiments complete
- [ ] Check final losses are reasonable
- [ ] Compare epsilon values across methods
- [ ] Validate log files generated
- [ ] Review summary.csv output

---

## Expected Outcomes After Fixes

### Baseline
- Loss: ~0.30-0.35
- Accuracy: ~88-92%
- Time: ~1 min

### DP (with tuning)
- Loss: ~0.35-0.45
- Accuracy: ~80-85%
- Time: ~2-3 min
- Epsilon: ~5-10

### ADP (with tuning)
- Loss: ~0.32-0.40
- Accuracy: ~83-88%
- Time: ~2-3 min
- Epsilon: ~3-7 (lower due to adaptive noise)

---

## Files Modified

1. `/home/samkit/Web3/BTP/client_adp.py` - Fixed privacy engine hook issue
   - Added privacy_engine instance tracking
   - Added model unwrapping logic
   - Fixed evaluation to use unwrapped model

---

## Next Steps

1. **Run the experiment again:**
   ```bash
   python experiment_runner.py
   ```

2. **Monitor the output** for:
   - No more "hooks twice" errors
   - All 3 experiments completing
   - Reasonable loss values

3. **If DP/ADP still underperform**, tune hyperparameters as suggested above

4. **Document final results** in your report with privacy-utility tradeoff analysis
