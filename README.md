# Astrophysics Rocket Navigation - Reinforcement Learning Environment

A custom OpenAI Gymnasium environment for training reinforcement learning agents to navigate a rocket through space, avoiding hazards and reaching target planets. The environment features realistic physics including gravity, fuel management, and various celestial objects.

## Overview

This project implements a complex RL environment where a rocket must navigate from Earth to one of several landable planets while:
- Managing limited fuel resources
- Avoiding black holes, suns, and asteroids
- Using gravitational forces strategically to save fuel
- Dealing with continuous state and action spaces

## Features

### Environment
- **Grid Size**: 1000x1000 continuous space
- **State Space**: 11-dimensional continuous (position, velocity, fuel, nearest object information)
- **Action Space**: 2-dimensional continuous (thrust magnitude [0-1], thrust angle [0-2π])
- **Physics**: Simplified gravity model with acceleration/deceleration
- **Celestial Objects**:
  - Earth (starting position at center)
  - 10 landable planets (terminal states with reward 0)
  - 120 non-landable planets (obstacles)
  - 5-8 suns (strong gravity, 50x50 grid influence)
  - 2-3 black holes (very strong gravity, 100x100 grid influence, event horizon)
  - 50-100 asteroids (random obstacles)

### Reward Structure
- Each action: -1 (encourages efficiency)
- Hitting asteroid: -2
- Within black hole influence: -2 per step
- Hitting sun or black hole center: -100 (terminal)
- Out of fuel: -2 per step
- Landing on valid planet: 0 (terminal, success)

### Environment Modes: Static vs Dynamic

The environment supports two modes to control how celestial objects are positioned:

**Static Environment (Default, Recommended for Training)**
- Celestial objects remain in the **same positions** across all episodes
- Agent can learn optimal paths for a specific configuration
- **Much easier to learn** - consistent training signal
- Faster convergence and better sample efficiency
- Ideal for initial training and algorithm development

```python
env = AstrophysicsEnv(seed=42, static_environment=True)  # Default
```

**Dynamic Environment (Advanced, For Generalization)**
- Celestial objects are **randomly repositioned** on each reset
- Agent must learn general navigation strategies
- **Much harder to learn** - inconsistent challenges each episode
- More robust and generalizable policies
- Use after agent masters static environment

```python
env = AstrophysicsEnv(seed=42, static_environment=False)
```

**Recommended Training Strategy:**
1. Train on static environment first (500-1000 episodes)
2. Fine-tune on dynamic environment for generalization (500+ episodes)
3. This curriculum learning approach provides both fast initial learning and robustness

### Implemented RL Algorithms

1. **Q-Learning with Discretization**
   - Tabular method with state/action space discretization
   - Epsilon-greedy exploration
   - Suitable for understanding basic RL concepts

2. **Deep Q-Network (DQN)**
   - Neural network-based Q-learning
   - Experience replay buffer
   - Target network for stability
   - Better handling of continuous state space

3. **Proximal Policy Optimization (PPO)**
   - Policy gradient method
   - Handles continuous actions directly
   - Actor-critic architecture with GAE
   - State-of-the-art performance

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd RL-sample

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- NumPy
- Gymnasium
- PyTorch
- Matplotlib

## Usage

### Training

Train a single algorithm (static environment - recommended):
```bash
# Q-Learning
python train.py --algorithm qlearning --episodes 500

# DQN
python train.py --algorithm dqn --episodes 500

# PPO (recommended - best performance)
python train.py --algorithm ppo --episodes 500
```

Train with dynamic environment (harder, more generalizable):
```bash
python train.py --algorithm ppo --episodes 500 --dynamic
```

Train all algorithms:
```bash
python train.py --algorithm all --episodes 500 --plot
```

Test the difference between static and dynamic:
```bash
python test_static_vs_dynamic.py
```

### Environment Visualization

**Static Environment View** - See the environment layout:
```bash
# Full visualization with gravity fields
python environment-test.py --seed 42

# Fast visualization without gravity computation
python environment-test.py --seed 42 --no-gravity

# Higher resolution gravity field (slower)
python environment-test.py --seed 42 --resolution 200

# Save to file
python environment-test.py --seed 42 --save environment_map.png

# Try different random seeds to see different configurations
python environment-test.py --seed 123
```

This will show you:
- **Objects Map**: All celestial objects with color coding and influence zones
- **Gravity Field**: Heat map showing gravitational force strength
- **Combined View**: Objects overlaid on gravity field
- **Statistics**: Environment configuration and object counts

**Agent Performance Visualization** - See how your trained agent performs:
```bash
# Visualize 10 runs of trained PPO agent
python run-visualization.py --algorithm ppo --runs 10

# Visualize with specific model
python run-visualization.py --algorithm dqn --model models/dqn_agent_final.pt --runs 20

# Save visualization
python run-visualization.py --algorithm ppo --runs 10 --save agent_performance.png
```

This will show you:
- **Individual Trajectories**: Each run's path (color-coded by outcome)
- **Average Trajectory**: Mean path with confidence bands
- **Position Heatmap**: Where the agent spends most time
- **Time Series**: Position, speed, fuel, and rewards over time
- **Statistics**: Success rate, average reward, and per-run details

### Evaluation

Evaluate a trained agent:
```bash
# Evaluate with visualization
python evaluate.py --algorithm ppo --episodes 10 --visualize

# Evaluate with live rendering
python evaluate.py --algorithm dqn --episodes 5 --render

# Compare all algorithms
python evaluate.py --algorithm all --episodes 10 --visualize
```

### Using the Environment Directly

```python
from astrophysics_env import AstrophysicsEnv

# Create environment
env = AstrophysicsEnv(
    grid_size=1000,
    max_fuel=500.0,
    max_steps=2000,
    render_mode='human',
    seed=42
)

# Reset environment
state, info = env.reset()

# Take actions
for _ in range(1000):
    # Random action: [thrust_magnitude, thrust_angle]
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        state, info = env.reset()
    else:
        state = next_state

env.close()
```

## Project Structure

```
RL-sample/
├── astrophysics_env.py          # Custom Gym environment
├── q_learning_agent.py          # Q-Learning implementation
├── dqn_agent.py                 # Deep Q-Network implementation
├── ppo_agent.py                 # PPO implementation
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── environment-test.py          # Environment visualization tool
├── run-visualization.py         # Agent performance visualization (NEW!)
├── test_environment.py          # Environment unit tests
├── test_static_vs_dynamic.py    # Static vs dynamic mode comparison
├── example_usage.py             # Usage examples
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── SUMMARY.md                   # Project summary
├── models/                      # Saved models (created during training)
└── results/                     # Training results and plots (created during training)
```

## Algorithm Details

### Q-Learning
- **Type**: Value-based, off-policy
- **State Space**: Discretized (20×20×10×10×10×10×8×10×8×10×8 bins)
- **Action Space**: Discretized (5 thrust levels × 8 angles)
- **Pros**: Simple, interpretable, guaranteed convergence
- **Cons**: Suffers from curse of dimensionality, requires discretization

### Deep Q-Network (DQN)
- **Type**: Value-based, off-policy
- **State Space**: Continuous (11 dimensions)
- **Action Space**: Discretized (7 thrust levels × 12 angles)
- **Key Features**:
  - Experience replay (buffer size: 50,000)
  - Target network (update frequency: 10)
  - Neural network: 3 hidden layers (256, 256, 128)
- **Pros**: Handles high-dimensional states, sample efficient
- **Cons**: Still requires action discretization

### Proximal Policy Optimization (PPO)
- **Type**: Policy gradient, on-policy
- **State Space**: Continuous (11 dimensions)
- **Action Space**: Continuous (2 dimensions)
- **Key Features**:
  - Actor-critic architecture
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Multiple epochs per update
- **Pros**: Direct continuous control, stable training, state-of-the-art
- **Cons**: Requires more samples, computationally intensive

## Physics and Dynamics

### Gravity Model
- **Suns**: Influence radius = 50 units, gravity strength = 0.5
- **Black Holes**: Influence radius = 100 units, gravity strength = 2.0
- Gravity force: `F = strength / (distance^2) * 100`
- Force decreases with square of distance (inverse square law)

### Rocket Dynamics
- Mass = 1 (simplified)
- Maximum thrust = 2.0
- Maximum velocity = 50 units/step
- Fuel consumption = 0.5 × thrust_magnitude per step
- State update: `velocity += (thrust_force + gravity_force) × 0.1`

### Collision Detection
- Asteroids: radius = 2 units
- Planets: radius = 3 units (landing zone = 1.5× radius)
- Suns: radius = 5 units
- Black holes: radius = 3 units, event horizon = 10 units

## Training Tips

1. **Start with simpler environments**: Reduce the number of celestial objects initially
2. **Adjust reward structure**: Tune reward values based on your goals
3. **Hyperparameter tuning**: Learning rate, epsilon decay, and network architecture significantly impact performance
4. **Curriculum learning**: Start with shorter episodes and gradually increase
5. **Monitor metrics**: Watch for:
   - Increasing average reward
   - Decreasing episode length (more efficient paths)
   - Increasing success rate (landing on valid planets)

## Customization

### Modify Environment Parameters
```python
env = AstrophysicsEnv(
    grid_size=500,          # Smaller grid
    max_fuel=1000.0,        # More fuel
    max_steps=1000,         # Fewer steps per episode
    seed=42                 # Reproducibility
)
```

### Adjust Celestial Objects
Edit `_generate_celestial_objects()` in `astrophysics_env.py` to change:
- Number of objects
- Positions
- Properties (gravity strength, influence radius)

### Change Reward Structure
Modify the `step()` method in `astrophysics_env.py` to adjust rewards.

## Visualization

The environment includes matplotlib-based rendering showing:
- Rocket trajectory (red)
- Earth (blue, starting position)
- Landable planets (green)
- Non-landable planets (gray)
- Suns (yellow, with influence zones)
- Black holes (black/purple, with event horizons)
- Asteroids (brown)
- Velocity vectors

## Performance Benchmarks

Expected performance after 500 training episodes (approximate):

| Algorithm | Mean Reward | Success Rate | Training Time |
|-----------|-------------|--------------|---------------|
| Q-Learning | -300 to -200 | 5-10% | Fast (~30 min) |
| DQN | -250 to -150 | 10-20% | Medium (~1-2 hours) |
| PPO | -200 to -100 | 20-35% | Slow (~2-4 hours) |

*Note: Results vary significantly based on hyperparameters and random seeds*

## Future Improvements

- [ ] Multi-objective optimization (fuel efficiency + time)
- [ ] Moving celestial objects
- [ ] Multiple rockets (multi-agent)
- [ ] More sophisticated physics (orbital mechanics)
- [ ] Curriculum learning implementation
- [ ] Distributed training support
- [ ] Additional RL algorithms (SAC, TD3, A3C)

## Troubleshooting

### Training is unstable
- Reduce learning rate
- Increase network size
- Adjust reward scaling
- Use reward clipping

### Agent doesn't learn
- Check reward structure (ensure positive rewards are achievable)
- Verify environment reset
- Increase exploration (higher epsilon)
- Reduce task complexity initially

### Out of memory errors
- Reduce batch size
- Reduce replay buffer size (DQN)
- Use smaller network
- Reduce update frequency

## Contributing

Contributions are welcome! Areas for improvement:
- Additional RL algorithms
- Better visualization
- Performance optimizations
- Documentation improvements
- Bug fixes

## License

MIT License - feel free to use this project for learning and research.

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{astrophysics_rl_env,
  title={Astrophysics Rocket Navigation RL Environment},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/RL-sample}}
}
```

## Acknowledgments

- Built with OpenAI Gymnasium
- Inspired by classic space navigation problems
- RL implementations based on standard algorithms from literature

## Contact

For questions or feedback, please open an issue on GitHub.

## Branch Information
gravityv2: Includes complex DQN + Newton Physics
gravityv1: Simple gravity implementation
v1
v2