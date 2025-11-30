# Model Saving on Interrupt - Updated

## What Changed

Updated `quick_train.py` and `train_ppo.py` to **automatically save the model** when you press Ctrl+C (KeyboardInterrupt).

---

## How It Works

### Before (Problem):
- Training runs for full 2M timesteps
- If you interrupt with Ctrl+C, NO model is saved
- You lose all progress 😢

### After (Fixed):
- Training can be interrupted at any time with Ctrl+C
- Model is **automatically saved** as `interrupted_model.zip`
- You keep all the learning progress! 🎉

---

## Files Modified

### 1. `train_ppo.py`
- Added `try/except KeyboardInterrupt` around `model.learn()`
- When interrupted:
  - Saves model to `checkpoints/sb3_ppo_quick/interrupted_model.zip`
  - Returns model and partial stats
  - Shows save confirmation message

### 2. `quick_train.py`
- Updated to handle the interrupt gracefully
- Shows user-friendly messages
- Cleans up properly on exit

---

## Usage

### Start Training:
```powershell
cd "C:\Users\chens\AI and Coding\Classes\AM158\RL-sample"
.\venv\Scripts\Activate.ps1
python gravitational/quick_train.py
```

### Interrupt When You See Good Results:
- Press **Ctrl+C** when you see:
  - Success rate increasing
  - Black hole rate decreasing  
  - Rewards improving

### What Happens:
```
⚠️  Training interrupted by user!
================================================================================
✓ Model saved to: checkpoints/sb3_ppo_quick/interrupted_model.zip
================================================================================
```

---

## Testing the Interrupted Model

### Option 1: Use test_sb3_policy.py (if it exists)
```powershell
python gravitational/test_sb3_policy.py checkpoints/sb3_ppo_quick/interrupted_model
```

### Option 2: Quick Python Script
```python
from stable_baselines3 import PPO
from gravitational_env import GravitationalDynamicsEnv

# Load the interrupted model
model = PPO.load("checkpoints/sb3_ppo_quick/interrupted_model")
env = GravitationalDynamicsEnv()

# Test it
state, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, truncated, info = env.step(action)
    if done:
        if info.get('terminal_reason') == 'success':
            print("✓ Reached target!")
        break
```

---

## Where Models Are Saved

After training (complete or interrupted), you'll find:

```
checkpoints/sb3_ppo_quick/
├── interrupted_model.zip      ← Saved on Ctrl+C
├── final_model.zip            ← Saved on completion
├── best_model.zip             ← Best performing model
├── ppo_checkpoint_50000_steps.zip
├── ppo_checkpoint_100000_steps.zip
└── logs/                      ← TensorBoard logs
```

---

## Tips

1. **Watch the output** for:
   - Success Rate increasing
   - Black Hole Rate decreasing
   - Average reward improving

2. **Interrupt when satisfied**:
   - Don't need to wait for full 2M timesteps
   - Can stop at 500K, 1M, etc.

3. **Test immediately**:
   ```powershell
   python gravitational/test_policy.py checkpoints/sb3_ppo_quick/interrupted_model 10
   ```

4. **Continue training later**:
   ```python
   model = PPO.load("checkpoints/sb3_ppo_quick/interrupted_model")
   model.learn(total_timesteps=1000000)  # Train for more
   model.save("checkpoints/sb3_ppo_quick/continued_model")
   ```

---

## Example Training Session

```
QUICK TRAIN - PPO with Stable-Baselines3
💡 Press Ctrl+C to stop training and save the current model
================================================================================

Episode 50 | Steps: 102400
  Success Rate (overall):    12.0%
  Black Hole Rate:           45.0%
  Avg Reward (recent):       -15.32

Episode 100 | Steps: 204800
  Success Rate (overall):    28.0%  ← Getting better!
  Black Hole Rate:           35.0%  ← Improving!
  Avg Reward (recent):       12.45  ← Positive rewards!

[Press Ctrl+C here]

⚠️  Training interrupted by user!
✓ Model saved to: checkpoints/sb3_ppo_quick/interrupted_model.zip
```

---

## Summary

✅ **Problem Fixed**: Model now saves on interrupt  
✅ **No Progress Lost**: All learning preserved  
✅ **Easy to Test**: Load and evaluate anytime  
✅ **Can Continue**: Resume training later if needed  

You're now in full control of your training! 🚀

