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

## Files

- `gravitational_env.py` - Main environment implementation
- `visualize_gravity.py` - Gravity field visualization tools
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
