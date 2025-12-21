# RLAMPSO Retraining Guide

**Date**: 2025-10-24
**Status**: 🔴 RETRAINING REQUIRED
**Reason**: Pretrained model incompatible with corrected code

---

## Why Retraining is Required

### The Problem

Your debug output shows parameters hitting extreme bounds:
```
w=0.1000, c1=0.0000, c2=3.0000  ← All at limits!
w=0.9000, c1=0.0000, c2=2.9971  ← Saturated!
```

**Normal parameters should be**: w=0.4-0.7, c1=1.5-2.5, c2=1.5-2.5

### Root Cause

1. **Pretrained model** was trained using ALL 20 action dimensions
2. **Our fix** only uses the FIRST 5 action dimensions (matching Python)
3. **Result**: The network's first 5 dimensions alone don't have enough information
4. **Output**: Saturated/extreme values that don't help PSO

**Analogy**: The network learned as a team of 20. We removed 15 members. The remaining 5 can't do the job alone.

### Evidence

- Parameters constantly hit 0.0, 0.1, 0.9, or 3.0 (bounds)
- No meaningful adaptation happening
- Scenario 1 result: Only 2.8% improvement (should be 0-5% in empty, but 20-40% in complex)
- This is NOT normal learning behavior

---

## Solution: You MUST Retrain

There are **no shortcuts**. The pretrained model fundamentally cannot work with the corrected code.

---

## Option 1: Online Learning (Temporary - Use Now)

**Status**: ✅ Already configured for you

I've set `rlamNoPretrained = ''` in your `run_comparison.m`

**How it works**:
- No pretrained weights
- Network starts with random initialization
- Learns DURING each UAV run
- Adapts to each specific scenario

**Expected results**:
- Scenario 1: ~0-5% improvement (both near optimal)
- Scenario 2-4: ~5-15% improvement (limited learning in 600 iterations)

**Pros**:
- ✅ Works immediately
- ✅ No training time needed
- ✅ Uses corrected code

**Cons**:
- ❌ Limited improvement (network doesn't have time to learn much)
- ❌ Not using pretrained knowledge
- ❌ Each run starts from scratch

**Use this while you prepare for full retraining.**

---

## Option 2: Full Retraining (Permanent - Do This)

### Training Strategy

Use the **two-phase approach** from the paper:

**Phase 1: CEC Pre-training** (General PSO Learning)
- Train on 28 CEC benchmark functions
- Each function: 100-400 episodes
- Learn general PSO parameter adaptation
- Time: 30-60 hours

**Phase 2: UAV Fine-tuning** (Domain-Specific)
- Transfer learning from Phase 1
- Train on UAV path planning
- 10× lower learning rates for stability
- 10,000 episodes
- Time: 10-40 hours

**Total time**: 40-100 hours (GPU highly recommended)

### Step-by-Step Instructions

#### Step 1: Prepare Environment

```matlab
cd('C:\Users\Thuan\Desktop\Copy_of_New Folder - Copy\algorithms\rlampso\matlabimplementation\training')
```

#### Step 2: Configure Training

Edit `train_baseline.m`:

```matlab
% Line 30-40: Training configuration
ENABLE_PHASE1_CEC = true;      % Enable CEC pre-training
ENABLE_PHASE2_UAV = true;      % Enable UAV fine-tuning
EXISTING_PRETRAINED_PATH = ''; % Start from scratch

% CEC pre-training settings
PHASE1_EPISODES_PER_FUNC = 100;  % 100 episodes per function (can increase to 400)
PHASE1_CEC_FUNCTIONS = 1:28;     % All 28 CEC functions

% UAV fine-tuning settings
PHASE2_EPISODES = 10000;         % UAV episodes (paper uses 10k)
PHASE2_LR_REDUCTION = 0.1;       % 10× lower learning rate
```

#### Step 3: Run Training

**WARNING**: This will take 40-100 hours!

```matlab
% Make sure you're in the right directory
cd('C:\Users\Thuan\Desktop\Copy_of_New Folder - Copy\algorithms\rlampso\matlabimplementation\training')

% Run training script
train_baseline
```

**What happens**:
1. **Phase 1** (30-60 hours):
   - Trains on CEC function 1 (100 episodes)
   - Saves checkpoint
   - Trains on CEC function 2 (100 episodes)
   - ... repeats for all 28 functions
   - Saves final CEC-pretrained model

2. **Phase 2** (10-40 hours):
   - Loads CEC-pretrained model
   - Fine-tunes on UAV path planning
   - Saves checkpoints every 1000 episodes
   - Saves final UAV-tuned model

3. **Output**:
   - Models saved in `training/models/`
   - Final model: `uav_baseline_YYYYMMDD_HHMMSS.mat`

#### Step 4: Use Trained Model

After training completes, update `run_comparison.m`:

```matlab
% Find your new model file (e.g., uav_baseline_20251025_120000.mat)
rlamNoPretrained = 'models/uav_baseline_20251025_120000.mat';
```

Then test:
```matlab
cd('C:\Users\Thuan\Desktop\Copy_of_New Folder - Copy')
run_comparison
```

### Expected Results After Retraining

| Scenario | PSO | RLAMPSO (Retrained) | Improvement |
|----------|-----|---------------------|-------------|
| 1 (0T0O) | ~1800 | ~1750-1800 | 0-5% |
| 2 (30T30O) | ~1834 | ~1300-1500 | **20-35%** |
| 3 (60T60O) | ~2100 | ~1400-1700 | **25-40%** |
| 4 (90T90O) | ~2400 | ~1500-1900 | **30-45%** |

---

## Option 3: Quick Retrain (Compromise)

If you don't have 100 hours, try this **fast training**:

### Modified Settings

```matlab
% In train_baseline.m:
ENABLE_PHASE1_CEC = true;
ENABLE_PHASE2_UAV = true;

% REDUCED training
PHASE1_EPISODES_PER_FUNC = 50;   % Reduced from 100 (faster)
PHASE1_CEC_FUNCTIONS = [1:10];   % Only 10 functions (faster)
PHASE2_EPISODES = 2000;          % Reduced from 10000 (faster)
```

**Time**: ~15-30 hours

**Expected performance**: 10-25% improvement (less than full training, but better than online learning)

**Use case**: You need reasonable results quickly

---

## Training Tips

### 1. Monitor Progress

Training prints progress every 10 episodes:
```
Episode 10/100: Reward=0.45, Actor Loss=0.12, Critic Loss=0.34
Episode 20/100: Reward=0.52, Actor Loss=0.10, Critic Loss=0.29
...
```

**Good signs**:
- Reward increasing over time
- Losses decreasing
- No NaN or Inf values

**Bad signs**:
- Reward stuck or decreasing
- Loss = NaN or Inf
- Very high losses (>100)

### 2. Use GPU

Training is **10-20× faster** on GPU!

Check if GPU available:
```matlab
canUseGPU()  % Should return true
```

If false, training will still work but be slower.

### 3. Save Checkpoints

The training script automatically saves:
- Every 1000 episodes
- After each CEC function
- At the end

**Don't interrupt training** - you'll lose progress!

### 4. Resume from Checkpoint

If training crashes, you can resume:

```matlab
% In train_baseline.m:
EXISTING_PRETRAINED_PATH = 'models/uav_baseline_20251025_checkpoint_5000.mat';
ENABLE_PHASE1_CEC = false;  % Skip CEC (already done)
ENABLE_PHASE2_UAV = true;   % Continue UAV training
```

---

## Troubleshooting

### Training is too slow

**Problem**: 1 episode takes >10 seconds

**Solutions**:
- Use GPU (10-20× faster)
- Reduce `PHASE1_EPISODES_PER_FUNC` to 50
- Reduce `PHASE2_EPISODES` to 2000
- Use fewer CEC functions (10 instead of 28)

### Loss becomes NaN

**Problem**: Training shows `Loss = NaN`

**Solutions**:
- Lower learning rates in `RLAMPSO_Config.m`:
  ```matlab
  initialActorLR = 0.00001;   % Already correct
  initialCriticLR = 0.0001;   % Reduce from 0.001 if NaN
  ```
- Add gradient clipping (already implemented)
- Check for invalid states/rewards

### Not enough memory

**Problem**: Out of memory error

**Solutions**:
- Reduce batch size:
  ```matlab
  config.batchSize = 128;  % Down from 256
  ```
- Reduce buffer size:
  ```matlab
  config.bufferSize = 500000;  % Down from 1000000
  ```
- Close other programs

### Results not improving

**Problem**: Trained model doesn't beat PSO

**Solutions**:
- Check parameters aren't saturated (w, c1, c2 in reasonable ranges)
- Verify all 4 bug fixes are applied
- Train longer (more episodes)
- Try different hyperparameters

---

## Verification Checklist

Before retraining, verify all fixes are applied:

- [ ] ✅ Architecture fix: Uses global parameters (avgW, avgC1, avgC2)
- [ ] ✅ Action fix: Uses only first 5 elements of action vector
- [ ] ✅ Clipping fix: Actions clipped to [-1, 1] after noise
- [ ] ✅ Critic LR fix: Set to 0.001 (not 0.0001)

**All should be checked!** Otherwise retraining will have same bugs.

---

## Timeline

| Task | Time | What You Get |
|------|------|-------------|
| **Online Learning (Now)** | 0 hours | 5-15% improvement, works immediately |
| **Quick Retrain** | 15-30 hours | 10-25% improvement |
| **Full Retrain (Recommended)** | 40-100 hours | 20-40% improvement |

---

## My Recommendation

### Immediate (Today)

✅ **Use online learning** (already configured)
- Test all 4 scenarios
- Expect 5-15% improvement
- This proves the fixes work

### Short-term (This Week)

⚠️ **Start quick retrain** (15-30 hours)
- Use reduced settings (10 CEC functions, 2000 UAV episodes)
- Get 10-25% improvement
- Good enough for most use cases

### Long-term (If Publishing/Production)

🎯 **Do full retrain** (40-100 hours)
- Use full settings (28 CEC functions, 10k UAV episodes)
- Get maximum 20-40% improvement
- Publication-quality results

---

## Summary

**Current Status**:
- ✅ All code fixes applied
- ❌ Pretrained model incompatible
- ⚠️ Currently using online learning (temporary)

**Next Steps**:
1. Test online learning on all scenarios (verify fixes work)
2. Start retraining (quick or full, your choice)
3. Replace online learning with retrained model
4. Achieve 20-40% improvement!

**Bottom Line**: You have correct code now, but need a model trained with that correct code to see full benefits.

---

Generated: 2025-10-24
Status: 🔴 RETRAINING REQUIRED
Temporary Solution: ✅ Online learning enabled
Next Action: Start retraining (quick or full)
