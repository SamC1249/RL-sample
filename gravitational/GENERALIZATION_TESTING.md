# Policy Generalization Testing

## The Question

**Does the agent learn a generalizable navigation strategy, or does it just memorize the specific layout?**

Since the current environment has **fixed positions** (seed=42):
- Black hole at (50, 50)
- Target at (85, 85)  
- Planets at fixed locations

The agent might just memorize: "go southeast, avoid center" rather than learning true spatial reasoning.

---

## How to Test Generalization

Use the new `test_generalization.py` script:

```powershell
# Activate venv
cd "C:\Users\chens\AI and Coding\Classes\AM158\RL-sample"
.\venv\Scripts\Activate.ps1

# Test your trained model
python gravitational/test_generalization.py checkpoints/sb3_ppo_quick/final_model

# Or test interrupted model
python gravitational/test_generalization.py checkpoints/sb3_ppo_quick/interrupted_model

# Customize number of tests
python gravitational/test_generalization.py checkpoints/sb3_ppo_quick/final_model 20 15
# (20 random configs, 15 episodes each)
```

---

## What It Tests

### Test 1: Original Configuration ✓
- Same layout the agent trained on
- **Expected: High success** (this is what it knows)

### Test 2: Randomized Configurations 🎲
- 10 different random seeds
- Black hole in different center positions
- Target in different corners
- Planets scattered differently
- **Expected: Reveals true generalization**

### Test 3: Different Starting Positions 📍
- Same layout, but start from:
  - (10, 90) - Top-left
  - (90, 10) - Bottom-right
  - (50, 10) - Bottom-center
  - (20, 50) - Left-center
- **Expected: Tests spatial flexibility**

---

## Example Output

```
================================================================================
POLICY GENERALIZATION TEST
================================================================================

TEST 1: Original Environment (What it trained on)
  Results for Original Config:
    Success Rate:    85.0%  ✓ Good on training config
    Black Hole Rate: 10.0%
    Avg Reward:      45.23

TEST 2: Randomized Configurations
  Config 1/10 (seed=100):
    Black hole: (45.3, 52.1)
    Target: (88.7, 10.5)
    Success: 42.0% | Black Hole: 35.0%  ⚠️ Performance drops!
  
  Config 2/10 (seed=101):
    Black hole: (58.2, 48.9)
    Target: (12.3, 85.7)
    Success: 38.0% | Black Hole: 40.0%
  ...

TEST 3: Different Starting Positions
  Start (10, 10): Success 85.0%  ✓ Original position
  Start (10, 90): Success 65.0%  ⚠️ Some drop
  Start (90, 10): Success 55.0%  ⚠️ Bigger drop
  Start (50, 10): Success 40.0%  ❌ Struggles here

================================================================================
GENERALIZATION SUMMARY
================================================================================

📊 Performance Comparison:

1. Original Training Config:
   Success Rate:    85.0%
   Avg Reward:      45.23

2. Random Configurations (Avg across 10 configs):
   Success Rate:    42.0%  ← 49% of original performance
   Avg Reward:      12.15

3. Different Start Positions:
   Success Rate:    61.3%  ← 72% of original performance
   Avg Reward:      28.45

📈 Generalization Analysis:
   Random Config Performance: 49.4% of original
   Different Start Performance: 72.1% of original

   ⚠️  MODERATE: Some generalization, but performance drops
      Agent partially learned spatial reasoning
```

---

## Interpreting Results

### ✅ GOOD Generalization (>70% of original performance)
- Agent learned robust navigation principles
- Understands gravity avoidance
- Can navigate varied layouts
- **Likely learned true spatial reasoning**

### ⚠️ MODERATE Generalization (40-70% of original)
- Agent has some transferable skills
- But also memorized specific patterns
- May navigate some new configs, fail others
- **Mix of learning and memorization**

### ❌ POOR Generalization (<40% of original)
- Agent heavily memorized training layout
- Specific route: "go this way, avoid that spot"
- Fails dramatically on new configurations
- **Mostly memorization, little true learning**

---

## Why This Matters

### With Good Generalization:
```
Agent learned: "Navigate toward target while avoiding strong gravity"
→ Works in any configuration
→ Real understanding of dynamics
```

### With Poor Generalization:
```
Agent learned: "From (10,10), move southeast while avoiding center"
→ Only works on training layout
→ Just memorized a route
```

---

## Expected Reality for Your Current Setup

Given that:
1. Training used **fixed seed** and positions
2. State space is **limited** ([x, y, vx, vy] only - no distance info)
3. Agent trained for relatively few episodes

**Likely outcome: MODERATE generalization**
- ~50-70% performance on random configs
- Agent learned some gravity avoidance
- But also memorized specific training layout
- Different starting positions will challenge it

---

## How to Improve Generalization

### Option 1: Train on Randomized Environments
Modify training to randomize positions each episode:

```python
# In train_ppo.py, modify make_env():
def make_env():
    def _init():
        env = GeneralizableGravEnv(
            randomize_positions=True,  # ← Randomize!
            grid_size=100, 
            G=1e-3, 
            k=1.0
        )
        env = Monitor(env)
        return env
    return _init
```

### Option 2: Curriculum Learning
Start with fixed layout, gradually increase randomization:
- Episodes 0-500K: Fixed layout (learn basics)
- Episodes 500K-1M: Slight randomization (±10 units)
- Episodes 1M+: Full randomization

### Option 3: Enhance State Space
Give agent more information (but this changes the problem):
- Add distance to target
- Add distance to black hole
- Makes generalization easier but less challenging

---

## Quick Test Right Now

Want to see current generalization? Run:

```powershell
python gravitational/test_generalization.py checkpoints/sb3_ppo_quick/best_model 5 20
```

This will test your current model on:
- 5 random configurations
- 20 episodes per config
- Takes ~5-10 minutes

You'll quickly see if your agent learned general navigation or memorized the specific layout!

---

## Summary

**The script reveals**:
- ✅ How well policy transfers to new scenarios
- ✅ Whether agent learned principles vs memorized
- ✅ Where the agent struggles (new starts, new layouts)

**Most RL agents trained on fixed environments**: **Moderate generalization**
- Good on similar configs
- Struggle on very different configs
- This is expected and normal!

To get excellent generalization, you'd need to train on diverse environments from the start. 🚀

