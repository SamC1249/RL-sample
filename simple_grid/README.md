# Simple Grid World - Policy Iteration

A simplified version of the astrophysics simulation for educational purposes and testing classic RL algorithms with gravitational dynamics.

## Environment Specifications

- **Grid Size**: 100×100
- **State Space**: 10,000 discrete states
- **Action Space**: 4 discrete actions (UP, DOWN, LEFT, RIGHT)
- **Stochastic**: Actions may be affected by gravitational forces

### Objects and Rewards

| Object | Position | Reward | Terminal |
|--------|----------|--------|----------|
| Black Hole | (20, 20) | -100 | Yes |
| Planet 1 | (40, 30) | -1 | No |
| Planet 2 | (60, 70) | -1 | No |
| Planet 3 | (30, 80) | -1 | No |
| Target | (80, 80) | 0 | Yes |
| Empty Cell | Any other | -1 | No |

### Gravitational Dynamics

#### Black Hole Gravity (Deterministic)
- **Effect**: If the agent is within a 2-block Manhattan distance of the black hole, it is **deterministically** pulled into the black hole
- **Radius**: 2 blocks (Manhattan distance)
- **Example**: At position (19, 20), the agent is 1 block away from the black hole at (20, 20), so any action will result in ending up in the black hole

#### Planet Gravity (Stochastic)
- **Effect**: If the agent is exactly 1 block away from a planet (adjacent) and takes an action that moves **away** from the planet, there is a **20% probability** that the agent will be pulled to the planet instead
- **Radius**: 1 block (Manhattan distance)
- **Probability**: 0.2 (20% chance of being pulled)
- **Example**: At position (40, 29), the agent is adjacent to planet at (40, 30). If the agent moves left (away from planet), there's a 20% chance it ends up at (40, 30) instead of (39, 29)

## Files

- `simple_env.py`: Grid world environment implementation
- `policy_iteration.py`: Policy iteration algorithm
- `run_policy_iteration.py`: Main script to execute and evaluate policy iteration

## Usage

Run policy iteration:

```bash
cd simple_grid
python run_policy_iteration.py
```

This will:
1. Create the 100×100 grid environment
2. Run policy iteration to find the optimal policy
3. Evaluate the learned policy over 100 episodes
4. Display results including success rate and sample trajectory

## Algorithm Details

**Policy Iteration** consists of two main steps:

1. **Policy Evaluation**: Compute the value function V(s) for the current policy using iterative evaluation, accounting for stochastic transitions due to gravity
2. **Policy Improvement**: Update the policy to be greedy with respect to V(s), considering expected values over all possible outcomes

The algorithm iterates these steps until the policy stabilizes (convergence).

### Parameters

- `gamma`: Discount factor (default: 0.99)
- `theta`: Convergence threshold for policy evaluation (default: 1e-6)
- `black_hole_radius`: Manhattan distance for black hole gravity (default: 2)
- `planet_radius`: Manhattan distance for planet gravity (default: 1)
- `planet_pull_prob`: Probability of being pulled by planet gravity (default: 0.2)

## Expected Results

The optimal policy should guide the agent from any starting position to the target at (80, 80) while:
- **Avoiding the black hole** danger zone (within 2 blocks of position (20, 20))
- **Navigating around planets** to minimize the risk of stochastic gravitational pull
- **Minimizing total expected cost** considering both rewards and gravitational risks

Due to gravitational dynamics, the policy will be more conservative near hazards and may take longer paths to ensure safety.
