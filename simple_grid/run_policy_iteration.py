#!/usr/bin/env python3
"""
Main script to run Policy Iteration on Simple Grid World

This script:
1. Creates a simple 100x100 grid environment
2. Runs policy iteration to find optimal policy
3. Evaluates the learned policy
4. Visualizes results
"""
import sys
import numpy as np
from simple_env import SimpleGridEnv
from policy_iteration import PolicyIteration


def main():
    """Main execution function."""
    print("=" * 80)
    print("SIMPLE GRID WORLD - POLICY ITERATION")
    print("=" * 80)

    # Create environment
    print("\nCreating environment...")
    env = SimpleGridEnv(grid_size=100)

    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Total states: {env.n_states}")
    print(f"Actions: {env.n_actions} (UP, DOWN, LEFT, RIGHT)")
    print(f"\nObjects:")
    print(f"  Black Hole: {env.black_hole_pos} (reward: -100)")
    print(f"  Planets: {env.planet_positions} (reward: -1 each)")
    print(f"  Target: {env.target_pos} (reward: 0)")
    print(f"  Default cell reward: -1")

    # Create policy iteration solver
    print("\n" + "=" * 80)
    gamma = 0.99
    theta = 1e-6

    solver = PolicyIteration(env, gamma=gamma, theta=theta)

    # Run policy iteration
    print("\nRunning Policy Iteration...")
    result = solver.solve(max_iterations=100, verbose=True)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"\nValue Function Statistics:")
    print(f"  Mean: {result['value_function'].mean():.4f}")
    print(f"  Std:  {result['value_function'].std():.4f}")
    print(f"  Min:  {result['value_function'].min():.4f}")
    print(f"  Max:  {result['value_function'].max():.4f}")

    # Visualize policy for subset
    solver.visualize_policy(show_range=(0, 30))

    # Evaluate policy
    print("\n" + "=" * 80)
    print("POLICY EVALUATION")
    print("=" * 80)
    print("\nEvaluating policy performance...")

    eval_results = solver.evaluate_policy(num_episodes=100, max_steps=1000)

    print(f"\nResults over {eval_results['total_episodes']} episodes:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {eval_results['mean_episode_length']:.2f}")
    print(f"  Success Rate: {eval_results['success_rate']*100:.2f}%")

    # Show sample trajectory
    print("\n" + "=" * 80)
    print("SAMPLE TRAJECTORY")
    print("=" * 80)

    state = env.reset(start_pos=(0, 0))
    trajectory = [env._state_to_pos(state)]
    total_reward = 0

    print(f"\nStarting from {env._state_to_pos(state)}")

    for step in range(200):  # Max 200 steps for display
        action = result['policy'][state]
        next_state, reward, done, info = env.step(action)

        trajectory.append(info['position'])
        total_reward += reward

        if done:
            print(f"Reached {'TARGET' if info['reached_target'] else 'BLACK HOLE'} at {info['position']}")
            print(f"Steps: {step + 1}")
            print(f"Total Reward: {total_reward:.2f}")
            break

        state = next_state

    # Show first and last few positions
    print(f"\nFirst 10 positions: {trajectory[:10]}")
    if len(trajectory) > 20:
        print(f"...")
        print(f"Last 10 positions: {trajectory[-10:]}")
    else:
        print(f"All positions: {trajectory}")

    print("\n" + "=" * 80)
    print("Policy Iteration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
