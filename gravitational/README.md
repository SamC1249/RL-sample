# Gravitational Dynamics Reinforcement Learning Environment

A realistic 100×100 grid world environment simulating Newtonian gravitational dynamics for reinforcement learning agents.

## Overview

This environment models a spaceship navigating through a gravitational field created by multiple celestial bodies. The agent must reach a target planet while avoiding black holes and minimizing fuel consumption.

## Environment Specifications

### Grid World
- **Size**: 100×100 grid
- **Gravitational Constant**: G = 10⁻³ (simplified for simulation)
- **Time Step**: Discrete with Δt = 1

### Agent (Spaceship)
- **Action Space**: Mixed discrete-continuous
  - Direction: Discrete {0: up, 1: right, 2: down, 3: left}
  - Thrust magnitude: Continuous T ∈ [0, 1]
- **Observation Space**: [x, y, vx, vy] (position and velocity)
- **Starting Position**: (10, 10)

### Celestial Bodies

| Type | Count | Mass | Reward | Event Horizon |
|------|-------|------|--------|---------------|
| Black Hole | 1 | 1×10⁸ | -100 | r = ⌊3 + log₁₀(m/10⁶)⌋ ≤ 6 |
| Normal Planet | 3 | 1×10⁵ | -1 | None |
| Target Planet | 1 | 1×10⁵ | 0 | None |

### Physics

#### Gravitational Field
Gravity follows Newton's law of universal gravitation:

```
g⃗(x,y) = -G Σᵢ (mᵢ / rᵢ²) r̂ᵢ
```

where:
- `G = 10⁻³` is the gravitational constant
- `mᵢ` is the mass of celestial body i
- `rᵢ` is the distance to body i
- `r̂ᵢ` is the unit vector pointing from body i to position (x,y)

The gravity field is pre-computed for all grid cells for efficiency.

#### Dynamics
```
v(t+1) = 0.95 × [v(t) + a_thrust + g(x,y)]
x(t+1) = x(t) + v(t+1)
```

Velocity damping (0.95) prevents unbounded growth.

### Reward Function

The reward at each step is:

```
r = -(||g⃗|| · (1 - cos θ)) + k·T
```

where:
- `||g⃗||` is the local gravity magnitude
- `θ` is the angle between thrust direction and gravity direction
- `k = 0.1` is the thrust reward coefficient
- `T` is the thrust magnitude

**Interpretation**:
- **Gravity cost**: Penalizes movement that doesn't oppose gravity
- **Thrust reward**: Small positive reward for using thrust efficiently
- Optimal strategy: Thrust opposite to gravity when gravity is strong

### Terminal Conditions

Episode ends when:
1. **Success**: Agent reaches target planet (reward = 0)
2. **Failure**: Agent enters black hole event horizon (reward = -100)
3. **Truncation**: Maximum steps (1000) reached

## Usage

### Basic Example

```python
from gravitational_env import GravitationalDynamicsEnv

# Create environment
env = GravitationalDynamicsEnv()

# Reset
obs, info = env.reset()

# Take action
action = [1, 0.5]  # direction=1 (right), thrust=0.5
obs, reward, terminated, truncated, info = env.step(action)
```

### Visualization

```python
from visualize_gravity import visualize_gravity_field, visualize_gravity_cross_sections

# Create environment
env = GravitationalDynamicsEnv()

# Visualize gravity field
visualize_gravity_field(env, save_path='gravity_field.png')

# Visualize cross-sections
visualize_gravity_cross_sections(env, save_path='gravity_cross_sections.png')
```

### Running Tests

```bash
# Test environment
python gravitational_env.py

# Generate visualizations
python visualize_gravity.py
```

## Training and Testing Convergence

### Quick Start

For a guided interactive experience:

```bash
python quick_start.py
```

Or run all tests non-interactively:

```bash
python quick_start.py --all
```

### Training an Agent

#### 1. Train DQN Agent

```bash
python train.py
```

This will:
- Train a DQN agent for up to 2000 episodes (or until convergence)
- Save checkpoints every 100 episodes to `checkpoints/dqn/`
- Generate training progress plots
- Track convergence metrics

**Convergence Criteria:**
- Success rate ≥ 70% over last 100 episodes
- Minimum 500 episodes completed

**Training Output:**
- `checkpoints/dqn/checkpoint_epXXX.pt` - Periodic checkpoints
- `checkpoints/dqn/final_model.pt` - Final trained model
- `checkpoints/dqn/training_stats.json` - Training statistics
- `checkpoints/dqn/training_progress.png` - 6-panel training visualization

#### 2. Custom Training Parameters

Edit `train.py` to customize:

```python
# Agent hyperparameters
agent = DQNAgent(
    state_dim=4,
    n_directions=4,
    n_thrust_levels=5,  # Discretization of thrust [0, 1]
    hidden_dims=[128, 128],  # Network architecture
    learning_rate=1e-3,
    gamma=0.99,  # Discount factor
    epsilon_start=1.0,  # Initial exploration
    epsilon_end=0.01,  # Final exploration
    epsilon_decay=0.995,  # Exploration decay rate
    buffer_capacity=10000,  # Replay buffer size
    batch_size=64
)

# Training parameters
n_episodes = 2000  # Maximum episodes
convergence_threshold = 0.7  # Success rate for convergence
```

#### 3. Train Different Agents

Modify `agent_type` in `train.py`:

```python
agent_type = "dqn"       # Deep Q-Network (learns from experience)
agent_type = "random"    # Random baseline (no learning)
agent_type = "heuristic" # Rule-based baseline
```

### Evaluating Agents

#### Compare All Agents

```bash
python evaluate.py
```

This will:
- Evaluate all available agents (Random, Heuristic, DQN)
- Run 100 test episodes per agent
- Generate comparison plots
- Visualize sample trajectories for each agent

**Evaluation Output:**
- `evaluation_results.json` - Detailed metrics
- `agent_comparison.png` - Performance comparison plot
- `trajectory_*.png` - Trajectory visualization for each agent

#### Evaluation Metrics

- **Success Rate**: Percentage of episodes reaching target planet
- **Average Reward**: Mean cumulative reward per episode
- **Average Length**: Mean episode duration (steps)
- **Black Hole Rate**: Percentage of episodes ending in black hole

### Monitoring Convergence

The training script automatically tracks convergence through multiple metrics:

#### 1. Success Rate
Primary convergence indicator. Training converges when success rate ≥ 70% over 100 episodes.

#### 2. Episode Rewards
Shows learning progress. Should increase over time as agent improves.

#### 3. Episode Length
Successful agents often complete episodes faster (shorter paths to target).

#### 4. Black Hole Death Rate
Should decrease as agent learns to avoid the black hole.

#### 5. Exploration Rate (Epsilon)
Decays from 1.0 to 0.01, showing transition from exploration to exploitation.

#### 6. Training Loss
Should decrease and stabilize as Q-network converges.

### Visualizing Training Progress

The training script automatically generates a 6-panel visualization:

1. **Episode Rewards** - Raw and moving average rewards
2. **Success Rate** - Moving window success rate with 70% target line
3. **Episode Length** - Steps per episode over time
4. **Black Hole Rate** - Death rate over time
5. **Training Loss** - Q-network loss
6. **Epsilon** - Exploration rate decay

```bash
# View training progress
open checkpoints/dqn/training_progress.png
```

### Expected Training Time

On a typical CPU:
- **Random Agent**: Instant (no training)
- **Heuristic Agent**: Instant (rule-based)
- **DQN Agent**: 10-30 minutes for convergence (500-1500 episodes)

GPU acceleration significantly reduces training time.

### Example Training Session

```bash
# Full training pipeline
cd gravitational/

# 1. Train DQN agent
python train.py

# Expected output:
# ================================================================================
# Episode 50/2000
# ================================================================================
#   Avg Reward (recent):          -45.32
#   Avg Length (recent):           245.12
#   Success Rate (recent):          12.00%
#   Black Hole Rate (recent):       35.00%
#   Current Epsilon:                 0.7788
# ...
# ================================================================================
# CONVERGED at episode 1247!
# Success rate: 72.45%
# ================================================================================

# 2. Evaluate trained agent
python evaluate.py

# Expected output:
# ================================================================================
# Comparing Agents
# ================================================================================
# Evaluating Random...
#   Success Rate: 2.00%
#   Avg Reward: -85.34 ± 45.23
#
# Evaluating Heuristic...
#   Success Rate: 45.00%
#   Avg Reward: -25.67 ± 30.12
#
# Evaluating DQN...
#   Success Rate: 73.00%
#   Avg Reward: -12.45 ± 15.34

# 3. View results
ls checkpoints/dqn/
# checkpoint_ep100.pt  checkpoint_ep200.pt  ...  final_model.pt
# training_progress.png  training_stats.json
```

### Troubleshooting

**Issue: Training not converging**
- Increase `n_episodes` to allow more training time
- Adjust `epsilon_decay` to balance exploration/exploitation
- Increase `n_thrust_levels` for finer thrust control
- Try different `hidden_dims` network architectures

**Issue: Agent consistently falls into black hole**
- Check black hole position (50, 50) and event horizon (radius 5)
- Heuristic agent should achieve ~45% success rate as baseline
- DQN needs sufficient exploration to learn avoidance

**Issue: Training is too slow**
- Reduce `hidden_dims` (e.g., [64, 64])
- Decrease `buffer_capacity` and `batch_size`
- Reduce `max_steps` per episode
- Use GPU acceleration with `device='cuda'`

## Files

- `gravitational_env.py` - Main environment implementation
- `visualize_gravity.py` - Gravity field visualization tools
- `dqn_agent.py` - DQN agent and baseline agents (Random, Heuristic)
- `train.py` - Training script with convergence tracking
- `evaluate.py` - Evaluation and trajectory visualization
- `quick_start.py` - Interactive quick start guide
- `README.md` - This file

## Key Features

1. **Realistic Physics**: Implements Newtonian gravity with multiple bodies
2. **Continuous-Discrete Hybrid**: Direction is discrete, thrust is continuous
3. **Pre-computed Gravity Field**: Efficient lookup with bilinear interpolation
4. **Rich Observations**: Includes position and velocity
5. **Challenging Navigation**: Must balance fuel efficiency with gravity avoidance

## Implementation Details

### Gravity Field Computation
The environment pre-computes the gravity gradient for all 10,000 grid cells at initialization. This provides:
- Fast lookups during simulation
- Smooth interpolation for continuous positions
- Consistent physics across the environment

### Black Hole Event Horizon
Event horizon radius calculation:
```python
r = min(floor(3 + log₁₀(mass / 10⁶)), 6)
```

For a black hole with mass 1×10⁸:
```
r = floor(3 + log₁₀(10⁸ / 10⁶)) = floor(3 + 2) = 5 units
```

### Reward Engineering
The reward function encourages:
- **Opposing gravity**: Minimizes (1 - cos θ) when thrust opposes gravity
- **Efficient thrust**: Small bonus for using thrust
- **Goal-seeking**: Terminal reward for reaching target

## Future Enhancements

Potential additions:
- Dynamic celestial bodies (orbiting planets)
- Fuel constraints
- Collision detection with planets
- Multiple target planets
- Time-varying gravity fields

## License

Part of the RL-sample repository.
