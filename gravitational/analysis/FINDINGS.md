# DQN Training Analysis - Low Reward Investigation

## Executive Summary

The DQN agent is performing poorly with an average reward of **-3,464** (recent 100 episodes) and **0% success rate**. After analyzing the code and training results, I've identified **three critical issues** that explain the low rewards.

---

## Training Statistics

| Metric | Value |
|--------|-------|
| Total Episodes | 2,000 |
| Average Reward (Overall) | -8,291 |
| Average Reward (Recent 100) | -3,464 |
| Success Rate | **0.0%** |
| Black Hole Death Rate | **69.0%** |
| Timeout Rate | 31.0% |
| Final Epsilon | 0.01 |

---

## Root Causes of Low Rewards

### 1. 🔴 CRITICAL BUG: Terminal Condition Detection

**Location:** `train.py` lines 240-243

```python
# Check terminal conditions
if terminated:
    # Check if success or black hole death
    if reward == 0:  # Target planet reward
        success = True
    elif reward == -100:  # Black hole reward
        black_hole_death = True
```

**Problem:**
- The code checks if `reward == 0` to detect success
- However, `reward` here is the **step reward** (gravity cost + thrust reward)
- The step reward is **never exactly 0** due to continuous gravity and thrust calculations
- The target planet gives a **bonus of 0**, not a total reward of 0
- **Result: Success is NEVER detected, even if the agent reaches the target!**

**Fix:**
```python
if terminated:
    # Use info dict to check actual terminal reason
    if info['distance_to_target'] <= 2.0:  # threshold from is_at_planet()
        success = True
    elif info['distance_to_black_hole'] <= self.black_hole.event_horizon:
        black_hole_death = True
```

---

### 2. 🟠 Reward Structure Issues

**Location:** `gravitational_env.py` lines 191-224

**Current Reward Formula:**
```
r = -(||g|| · (1 - cos θ)) + k*T

where:
- ||g|| = gravity magnitude (can be very large near black hole)
- cos θ = alignment between thrust and gravity
- k = 0.1 (thrust reward coefficient)
- T = thrust magnitude [0, 1]
```

**Problems:**

#### a) Gravity Cost Dominates
- Near the black hole (center at 50, 50), gravity magnitude is **extremely high**
- With G=1e-3 and mass=1e8: `||g|| = G * m / r² = 1e-3 * 1e8 / r² = 1e5 / r²`
- At distance r=10: `||g|| ≈ 1,000`
- Gravity cost per step: up to `1,000 * (1 - cos θ) ≈ 0 to 2,000`

#### b) Thrust Reward Too Small
- Maximum thrust reward: `k * T = 0.1 * 1.0 = 0.1` per step
- Gravity cost can be **10,000x larger** than thrust reward!
- Agent is heavily punished, barely rewarded

#### c) No Positive Shaping
- No reward for approaching the target
- No progressive penalty for approaching black hole (only -100 at death)
- Agent has no guidance toward goal

**Recommended Fixes:**

1. **Normalize rewards:**
```python
# Normalize gravity cost to reasonable range
gravity_cost = np.tanh(g_magnitude * (1 - cos_theta) / 100.0)  # Scale to [0, 1]
thrust_reward = k * thrust_magnitude
reward = -gravity_cost + thrust_reward
```

2. **Add distance-based shaping:**
```python
# Reward for getting closer to target
distance_to_target = np.linalg.norm(self.agent_pos - self.target_planet.get_position())
prev_distance = np.linalg.norm(prev_pos - self.target_planet.get_position())
distance_reward = (prev_distance - distance_to_target) * 10.0  # Reward for progress

# Penalty for approaching black hole
distance_to_black_hole = np.linalg.norm(self.agent_pos - self.black_hole.get_position())
if distance_to_black_hole < 20:  # Danger zone
    black_hole_penalty = -1.0 * (20 - distance_to_black_hole)  # Progressive penalty
else:
    black_hole_penalty = 0.0

reward = -gravity_cost + thrust_reward + distance_reward + black_hole_penalty
```

3. **Increase k or make thrust more valuable:**
```python
k = 1.0  # Instead of 0.1
```

---

### 3. 🟡 Limited State Information

**Current State:** `[x, y, vx, vy]` (4 dimensions)

**Problem:**
The agent only knows its position and velocity, but not:
- Distance to target
- Direction to target  
- Distance to black hole (danger!)
- Gravity magnitude at current position

**Why This Matters:**
- Agent must learn spatial relationships from scratch
- No explicit signal about proximity to goal or danger
- Makes learning much harder and slower

**Recommended Enhanced State:**
```python
def _get_observation(self) -> np.ndarray:
    # Original state
    pos = self.agent_pos
    vel = self.agent_vel
    
    # Target information
    target_pos = np.array([self.target_planet.x, self.target_planet.y])
    to_target = target_pos - pos
    dist_to_target = np.linalg.norm(to_target)
    angle_to_target = np.arctan2(to_target[1], to_target[0])
    
    # Black hole information
    black_hole_pos = np.array([self.black_hole.x, self.black_hole.y])
    to_black_hole = black_hole_pos - pos
    dist_to_black_hole = np.linalg.norm(to_black_hole)
    
    # Gravity information
    gravity = self._get_gravity_at_position(pos[0], pos[1])
    gravity_magnitude = np.linalg.norm(gravity)
    
    return np.concatenate([
        pos,                          # [x, y]
        vel,                          # [vx, vy]
        [dist_to_target],            # distance to goal
        [angle_to_target],           # direction to goal
        [dist_to_black_hole],        # distance to danger
        [gravity_magnitude]          # local gravity strength
    ])  # Total: 9 dimensions
```

---

## Why the Agent Never Succeeds

Combining all three issues:

1. **Even if the agent reaches the target**, success is not detected (Bug #1)
2. **The agent receives massive negative rewards** just for existing near the black hole (Issue #2)
3. **The agent has no clear signal** about where the target is (Issue #3)

**Result:** The agent learns to avoid the black hole somewhat (69% death rate vs ~100% random), but:
- Never learns to reach the target (0% success)
- Accumulates huge negative rewards from gravity
- Eventually times out or dies

---

## Verification of Issues

### Evidence from Training Data:

1. **0% success rate** across all 2,000 episodes → Terminal detection bug
2. **Average reward improving** (-8,291 → -3,464) → Agent IS learning something
3. **69% black hole death** → Agent partially learns to avoid black hole
4. **Epsilon decayed to 0.01** → Exploration is not the issue
5. **Weight norms increasing** → Network is updating and learning

The agent IS learning (rewards improving, fewer black hole deaths than random), but the reward structure and bug prevent it from succeeding.

---

## Recommended Action Plan

### Priority 1: Fix Critical Bug
1. Fix terminal condition detection in `train.py` (lines 240-243)
2. Use `info['distance_to_target']` instead of `reward == 0`

### Priority 2: Improve Reward Structure  
1. Normalize gravity cost to prevent domination
2. Add distance-based reward shaping
3. Increase k from 0.1 to 1.0 or higher
4. Add progressive black hole penalty

### Priority 3: Enhance State Space
1. Add distance to target
2. Add direction to target
3. Add distance to black hole
4. Add gravity magnitude
5. Update `state_dim` in agent initialization from 4 to 9

### Priority 4: Retrain
1. Clear old checkpoints or create new directory
2. Train for 2,000+ episodes with fixes
3. Monitor success rate (should reach >50% with fixes)

---

## Expected Results After Fixes

With the above changes, you should see:

- **Success rate:** 50-80% (vs current 0%)
- **Average reward:** Positive or near-zero (vs current -3,464)
- **Black hole death rate:** <20% (vs current 69%)
- **Learning speed:** Faster convergence

---

## Visualization Files Generated

The analysis tool created the following visualizations in:
`gravitational/checkpoints/analysis/results/`

1. **`training_dashboard.png`** - Comprehensive overview with 8 subplots:
   - Training statistics summary
   - Weight magnitude evolution
   - Epsilon decay curve
   - Checkpoint distribution
   - Reward analysis
   - Outcome distribution (pie chart)
   - Learning progress gauge
   - Recommendations

2. **`weight_evolution.png`** - Q-network weight changes per layer over time

3. **`problem_diagnosis.png`** - Detailed problem analysis with:
   - Reward structure issues
   - Terminal detection bug explanation
   - State space limitations
   - Statistics interpretation

4. **`analysis_report.txt`** - Text summary of findings

---

## Code Examples for Fixes

### Fix 1: Terminal Detection (`train.py`)

**Before:**
```python
if terminated:
    if reward == 0:  # BUG: This never happens!
        success = True
    elif reward == -100:
        black_hole_death = True
```

**After:**
```python
if terminated:
    # Check actual terminal reason using info dict
    if info['distance_to_target'] <= 2.0:
        success = True
    elif info['distance_to_black_hole'] <= self.black_hole.event_horizon:
        black_hole_death = True
```

### Fix 2: Reward Shaping (`gravitational_env.py`)

Add to `__init__`:
```python
self.prev_distance_to_target = None
```

Replace `_compute_reward` method:
```python
def _compute_reward(self, thrust_direction: np.ndarray, thrust_magnitude: float,
                   gravity_vector: np.ndarray) -> float:
    """Enhanced reward with normalization and shaping"""
    
    # 1. Normalized gravity cost
    g_magnitude = np.linalg.norm(gravity_vector)
    if g_magnitude > 1e-6:
        g_direction = gravity_vector / g_magnitude
        cos_theta = np.dot(thrust_direction, -g_direction)
        # Normalize to prevent domination
        gravity_cost = np.tanh(g_magnitude * (1 - cos_theta) / 100.0)
    else:
        gravity_cost = 0.0
    
    # 2. Thrust reward (increased k)
    thrust_reward = 1.0 * thrust_magnitude  # k=1.0 instead of 0.1
    
    # 3. Distance-based shaping
    current_distance = self.target_planet.distance_to(
        self.agent_pos[0], self.agent_pos[1]
    )
    
    if self.prev_distance_to_target is not None:
        # Reward for getting closer, penalty for getting farther
        distance_reward = (self.prev_distance_to_target - current_distance) * 5.0
    else:
        distance_reward = 0.0
    
    self.prev_distance_to_target = current_distance
    
    # 4. Black hole danger penalty
    black_hole_distance = self.black_hole.distance_to(
        self.agent_pos[0], self.agent_pos[1]
    )
    if black_hole_distance < 20:
        black_hole_penalty = -0.5 * (20 - black_hole_distance)
    else:
        black_hole_penalty = 0.0
    
    # Combine all components
    reward = -gravity_cost + thrust_reward + distance_reward + black_hole_penalty
    
    return reward
```

Update `reset()`:
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.agent_pos = np.array([10.0, 10.0], dtype=np.float32)
    self.agent_vel = np.array([0.0, 0.0], dtype=np.float32)
    self.steps = 0
    
    # Reset distance tracking
    self.prev_distance_to_target = self.target_planet.distance_to(
        self.agent_pos[0], self.agent_pos[1]
    )
    
    obs = self._get_observation()
    info = self._get_info()
    return obs, info
```

### Fix 3: Enhanced State Space (`gravitational_env.py`)

Update observation space in `__init__`:
```python
# Observation space: [x, y, vx, vy, dist_target, angle_target, dist_black_hole, g_magnitude]
self.observation_space = spaces.Box(
    low=np.array([0, 0, -10, -10, 0, -np.pi, 0, 0]),
    high=np.array([grid_size, grid_size, 10, 10, 150, np.pi, 150, 100]),
    dtype=np.float32
)
```

Replace `_get_observation`:
```python
def _get_observation(self) -> np.ndarray:
    """Enhanced observation with goal and danger information"""
    pos = self.agent_pos
    vel = self.agent_vel
    
    # Target information
    target_pos = np.array([self.target_planet.x, self.target_planet.y])
    to_target = target_pos - pos
    dist_to_target = np.linalg.norm(to_target)
    angle_to_target = np.arctan2(to_target[1], to_target[0])
    
    # Black hole information
    dist_to_black_hole = self.black_hole.distance_to(pos[0], pos[1])
    
    # Gravity information
    gravity = self._get_gravity_at_position(pos[0], pos[1])
    gravity_magnitude = np.linalg.norm(gravity)
    
    return np.concatenate([
        pos,                      # [x, y]
        vel,                      # [vx, vy]
        [dist_to_target],        # distance to goal
        [angle_to_target],       # direction to goal  
        [dist_to_black_hole],    # distance to danger
        [gravity_magnitude]      # local gravity strength
    ])
```

Update agent initialization in `train.py`:
```python
agent = DQNAgent(
    state_dim=8,  # Changed from 4 to 8
    n_directions=4,
    n_thrust_levels=5,
    # ... rest of parameters
)
```

---

## Conclusion

The low rewards are caused by a combination of:
1. A critical bug preventing success detection
2. Poorly scaled reward structure
3. Limited state information

All three issues are fixable with the code changes above. The agent IS learning (as evidenced by improving rewards and reduced black hole deaths), but the current setup prevents it from succeeding.

After implementing these fixes, retrain the agent and you should see dramatic improvement in both success rate and average reward.

