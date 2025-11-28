# Quick Summary: Why Rewards Are So Low

## The Numbers
- **Average Reward**: -3,464 (recent 100 episodes)
- **Success Rate**: 0%
- **Black Hole Deaths**: 69%
- **Episodes Trained**: 2,000

## The Problems (In Order of Severity)

### 🔴 Problem 1: Critical Bug in `train.py`
**Line 240-243**: Success detection is broken!

```python
if reward == 0:  # This NEVER happens!
    success = True
```

The code checks if step reward equals 0, but step reward is gravity + thrust, never exactly 0.

**Fix**: Check `info['distance_to_target'] <= 2.0` instead.

---

### 🟠 Problem 2: Reward Structure in `gravitational_env.py`
**Lines 191-224**: Rewards are massively imbalanced!

- **Gravity cost**: Can be -2,000 per step near black hole
- **Thrust reward**: Maximum +0.1 per step
- **Ratio**: Gravity is 20,000x stronger than thrust!

**Fix**: 
1. Normalize gravity cost: `np.tanh(gravity_cost / 100)`
2. Increase k from 0.1 to 1.0
3. Add distance-based shaping: reward getting closer to target

---

### 🟡 Problem 3: Limited State Space
**Current state**: `[x, y, vx, vy]` (4 values)

Agent doesn't know:
- How far to target
- Which direction is target
- How close to black hole (danger!)

**Fix**: Add `[dist_to_target, angle_to_target, dist_to_black_hole, gravity_magnitude]`

---

## Why This Explains Everything

1. Agent learns to avoid black hole somewhat (69% death vs 100% random)
2. But never reaches target (0% success) because:
   - Even if it did, success wouldn't be detected (Bug #1)
   - Massive negative rewards discourage exploration (Problem #2)
   - No clear signal about where target is (Problem #3)
3. Result: Huge negative cumulative rewards

---

## What To Do

1. **Read**: `FINDINGS.md` for detailed analysis and code fixes
2. **View**: Generated visualizations in `checkpoints/analysis/results/`
3. **Fix**: Implement the three fixes above
4. **Retrain**: Should see 50-80% success rate and positive rewards

---

## Evidence Agent IS Learning

Despite poor results, the agent IS learning:
- Rewards improved from -8,291 to -3,464
- Black hole deaths decreased (learning to avoid somewhat)
- Weight norms increasing (network updating)
- Epsilon decayed properly (exploration → exploitation)

The problems are **fixable** - the agent just needs better reward structure and bug fixes!

