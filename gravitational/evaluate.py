"""
Evaluation script for trained agents on Gravitational Dynamics Environment
Includes trajectory visualization and performance metrics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from gravitational_env import GravitationalDynamicsEnv
from dqn_agent import DQNAgent, RandomAgent, HeuristicAgent
from typing import List, Tuple, Dict
import json
import os


def evaluate_agent(
    agent,
    env: GravitationalDynamicsEnv,
    n_episodes: int = 100,
    max_steps: int = 1000,
    verbose: bool = True
) -> Dict:
    """
    Evaluate a trained agent.

    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    black_hole_count = 0
    timeout_count = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        success = False
        black_hole_death = False

        for step in range(max_steps):
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            state = next_state

            if terminated:
                if reward == 0:
                    success = True
                    success_count += 1
                elif reward == -100:
                    black_hole_death = True
                    black_hole_count += 1
                break

            if truncated:
                timeout_count += 1
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={step + 1}, "
                  f"Success={success}, "
                  f"Black Hole={black_hole_death}")

    metrics = {
        'n_episodes': n_episodes,
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'avg_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'success_rate': success_count / n_episodes,
        'black_hole_rate': black_hole_count / n_episodes,
        'timeout_rate': timeout_count / n_episodes,
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards))
    }

    return metrics


def record_trajectory(
    agent,
    env: GravitationalDynamicsEnv,
    max_steps: int = 1000
) -> Tuple[List, List, List, bool, bool]:
    """
    Record a single trajectory.

    Returns:
        positions, actions, rewards, success, black_hole_death
    """
    state, _ = env.reset()
    positions = [state[:2].copy()]
    actions = []
    rewards = []
    success = False
    black_hole_death = False

    for step in range(max_steps):
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)

        positions.append(next_state[:2].copy())
        actions.append(action.copy())
        rewards.append(reward)

        state = next_state

        if terminated:
            if reward == 0:
                success = True
            elif reward == -100:
                black_hole_death = True
            break

        if truncated:
            break

    return positions, actions, rewards, success, black_hole_death


def visualize_trajectory(
    env: GravitationalDynamicsEnv,
    positions: List[np.ndarray],
    actions: List[np.ndarray],
    rewards: List[float],
    success: bool,
    black_hole_death: bool,
    save_path: str = "trajectory.png"
):
    """Visualize agent trajectory on the gravity field"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Get gravity field for background
    gravity_field = env.get_gravity_field()
    gravity_magnitude = np.sqrt(gravity_field[:, :, 0]**2 + gravity_field[:, :, 1]**2)

    # Plot 1: Trajectory on gravity magnitude field
    ax1 = axes[0]
    im = ax1.imshow(gravity_magnitude.T, origin='lower', cmap='hot',
                   alpha=0.6, interpolation='bilinear')
    plt.colorbar(im, ax=ax1, label='Gravity Magnitude', fraction=0.046)

    # Plot trajectory
    positions_array = np.array(positions)
    ax1.plot(positions_array[:, 0], positions_array[:, 1],
            'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax1.scatter(positions_array[0, 0], positions_array[0, 1],
               c='green', s=200, marker='o', edgecolors='black',
               linewidths=2, label='Start', zorder=10)
    ax1.scatter(positions_array[-1, 0], positions_array[-1, 1],
               c='red' if black_hole_death else 'lime',
               s=200, marker='X', edgecolors='black',
               linewidths=2, label='End', zorder=10)

    # Add celestial bodies
    for body in env.celestial_bodies:
        if body.type == 'black_hole':
            color = 'black'
            marker = 'o'
            size = 300
            # Draw event horizon
            circle = Circle((body.x, body.y), body.event_horizon,
                          fill=False, edgecolor='white', linewidth=2,
                          linestyle='--', zorder=5)
            ax1.add_patch(circle)
        elif body.type == 'target_planet':
            color = 'lime'
            marker = '*'
            size = 400
        else:
            color = 'orange'
            marker = 'o'
            size = 200

        ax1.scatter(body.x, body.y, c=color, marker=marker, s=size,
                   edgecolors='black', linewidths=2, zorder=10)

    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title(f'Agent Trajectory ({"SUCCESS" if success else "BLACK HOLE" if black_hole_death else "TIMEOUT"})',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: Rewards and actions over time
    ax2 = axes[1]

    # Rewards
    ax2_reward = ax2.twinx()
    steps = list(range(len(rewards)))

    ax2.plot(steps, rewards, 'b-', linewidth=2, label='Reward', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12, color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(alpha=0.3)

    # Thrust magnitude
    if len(actions) > 0:
        thrust_values = [action[1] for action in actions]
        ax2_reward.plot(steps, thrust_values, 'r-', linewidth=2,
                       label='Thrust', alpha=0.7)
        ax2_reward.set_ylabel('Thrust Magnitude', fontsize=12, color='r')
        ax2_reward.tick_params(axis='y', labelcolor='r')
        ax2_reward.set_ylim([0, 1.1])

    ax2.set_title('Rewards and Actions Over Time', fontsize=14, fontweight='bold')

    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_reward.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.suptitle(f'Agent Trajectory Visualization (Steps: {len(positions)}, '
                f'Total Reward: {sum(rewards):.2f})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory visualization saved to: {save_path}")
    plt.close()


def compare_agents(
    env: GravitationalDynamicsEnv,
    agents_dict: Dict,
    n_episodes: int = 100,
    save_path: str = "agent_comparison.png"
):
    """Compare multiple agents"""
    results = {}

    print("="*80)
    print("Comparing Agents")
    print("="*80)

    for name, agent in agents_dict.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_agent(agent, env, n_episodes=n_episodes, verbose=False)
        results[name] = metrics

        print(f"  Success Rate: {metrics['success_rate']:.2%}")
        print(f"  Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Avg Length: {metrics['avg_length']:.1f} ± {metrics['std_length']:.1f}")
        print(f"  Black Hole Rate: {metrics['black_hole_rate']:.2%}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    agent_names = list(results.keys())
    metrics_to_plot = [
        ('success_rate', 'Success Rate', axes[0, 0]),
        ('avg_reward', 'Average Reward', axes[0, 1]),
        ('avg_length', 'Average Episode Length', axes[1, 0]),
        ('black_hole_rate', 'Black Hole Death Rate', axes[1, 1])
    ]

    for metric_key, metric_name, ax in metrics_to_plot:
        values = [results[name][metric_key] for name in agent_names]

        bars = ax.bar(agent_names, values, color=['steelblue', 'orange', 'green'][:len(agent_names)])
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}' if metric_key != 'success_rate' and metric_key != 'black_hole_rate'
                   else f'{height:.1%}',
                   ha='center', va='bottom', fontsize=10)

    plt.suptitle('Agent Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()

    return results


if __name__ == "__main__":
    print("="*80)
    print("Evaluating Agents on Gravitational Dynamics Environment")
    print("="*80)

    # Create environment
    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=0.1)

    # Create agents
    agents = {
        'Random': RandomAgent(),
        'Heuristic': HeuristicAgent(env)
    }

    # Check if DQN checkpoint exists
    dqn_checkpoint = "checkpoints/dqn/final_model.pt"
    if os.path.exists(dqn_checkpoint):
        print(f"\nLoading DQN agent from {dqn_checkpoint}")
        dqn_agent = DQNAgent(
            state_dim=4,
            n_directions=4,
            n_thrust_levels=5,
            hidden_dims=[128, 128]
        )
        dqn_agent.load(dqn_checkpoint)
        agents['DQN'] = dqn_agent
    else:
        print(f"\nDQN checkpoint not found: {dqn_checkpoint}")
        print("Training DQN for comparison...")
        dqn_agent = DQNAgent(
            state_dim=4,
            n_directions=4,
            n_thrust_levels=5,
            hidden_dims=[128, 128],
            epsilon_start=0.5,  # Start with some exploration
            epsilon_end=0.01,
            epsilon_decay=0.99
        )
        agents['DQN (untrained)'] = dqn_agent

    # Compare agents
    print("\n" + "="*80)
    print("Running Comparison")
    print("="*80)

    results = compare_agents(env, agents, n_episodes=100,
                            save_path="agent_comparison.png")

    # Save results
    with open("evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to: evaluation_results.json")

    # Visualize trajectories for each agent
    print("\n" + "="*80)
    print("Recording Sample Trajectories")
    print("="*80)

    for name, agent in agents.items():
        print(f"\nRecording trajectory for {name}...")
        positions, actions, rewards, success, black_hole = record_trajectory(
            agent, env, max_steps=1000
        )

        filename = f"trajectory_{name.lower().replace(' ', '_')}.png"
        visualize_trajectory(env, positions, actions, rewards, success,
                           black_hole, save_path=filename)

        print(f"  Total Steps: {len(positions)}")
        print(f"  Total Reward: {sum(rewards):.2f}")
        print(f"  Success: {success}")
        print(f"  Black Hole Death: {black_hole}")

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
