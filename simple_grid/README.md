# Simple Grid World - Policy Iteration

A simplified version of the astrophysics simulation for educational purposes and testing classic RL algorithms.

## Environment Specifications

- **Grid Size**: 100×100
- **State Space**: 10,000 discrete states
- **Action Space**: 4 discrete actions (UP, DOWN, LEFT, RIGHT)
- **Deterministic**: Actions always succeed (no stochasticity)

### Objects and Rewards

| Object | Position | Reward | Terminal |
|--------|----------|--------|----------|
| Black Hole | (20, 20) | -100 | Yes |
| Planet 1 | (40, 30) | -1 | No |
| Planet 2 | (60, 70) | -1 | No |
| Planet 3 | (30, 80) | -1 | No |
| Target | (80, 80) | 0 | Yes |
| Empty Cell | Any other | -1 | No |

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

1. **Policy Evaluation**: Compute the value function V(s) for the current policy using iterative evaluation
2. **Policy Improvement**: Update the policy to be greedy with respect to V(s)

The algorithm iterates these steps until the policy stabilizes (convergence).

### Parameters

- `gamma`: Discount factor (default: 0.99)
- `theta`: Convergence threshold for policy evaluation (default: 1e-6)

## Expected Results

The optimal policy should guide the agent from any starting position to the target at (80, 80) while avoiding the black hole at (20, 20). The success rate should approach 100% from most starting positions.
