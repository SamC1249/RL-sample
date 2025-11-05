"""
Rocket Run Visualization Tool
Visualizes agent performance with trajectory plots and time series analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import argparse
import os
import pickle
import torch
from typing import List, Dict, Tuple

from astrophysics_env import AstrophysicsEnv
from q_learning_agent import DiscretizedQLearningAgent
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent


def load_agent(algorithm: str, model_path: str):
    """Load a trained agent"""
    if algorithm.lower() == 'qlearning':
        agent = DiscretizedQLearningAgent()
        agent.load(model_path)
    elif algorithm.lower() == 'dqn':
        agent = DQNAgent()
        agent.load(model_path)
    elif algorithm.lower() == 'ppo':
        agent = PPOAgent()
        agent.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return agent


def run_episode(env, agent, algorithm: str, max_steps: int = 500) -> Dict:
    """
    Run a single episode and collect trajectory data

    Returns:
        Dictionary with trajectory data
    """
    state, info = env.reset()

    trajectory = {
        'positions': [info['rocket_pos'].copy()],
        'velocities': [info['rocket_vel'].copy()],
        'speeds': [np.linalg.norm(info['rocket_vel'])],
        'fuel': [info['fuel']],
        'actions': [],
        'rewards': [],
        'cumulative_rewards': [0],
        'steps': 0,
        'success': False,
        'termination_reason': 'truncated'
    }

    done = False
    cumulative_reward = 0

    while not done and trajectory['steps'] < max_steps:
        # Get action
        if algorithm.lower() == 'qlearning':
            action = agent.get_action(state, training=False)
        elif algorithm.lower() == 'dqn':
            action, _ = agent.get_action(state, training=False)
        elif algorithm.lower() == 'ppo':
            action, _, _ = agent.get_action(state, training=False)

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record data
        trajectory['positions'].append(info['rocket_pos'].copy())
        trajectory['velocities'].append(info['rocket_vel'].copy())
        trajectory['speeds'].append(np.linalg.norm(info['rocket_vel']))
        trajectory['fuel'].append(info['fuel'])
        trajectory['actions'].append(action.copy())
        trajectory['rewards'].append(reward)
        cumulative_reward += reward
        trajectory['cumulative_rewards'].append(cumulative_reward)
        trajectory['steps'] += 1

        state = next_state

        # Check termination reason
        if terminated:
            if reward >= -1:  # Successful landing
                trajectory['success'] = True
                trajectory['termination_reason'] = 'landing'
            elif reward <= -50:  # Hit sun or black hole
                trajectory['termination_reason'] = 'collision'
            else:
                trajectory['termination_reason'] = 'other'

    # Convert to numpy arrays
    trajectory['positions'] = np.array(trajectory['positions'])
    trajectory['velocities'] = np.array(trajectory['velocities'])
    trajectory['speeds'] = np.array(trajectory['speeds'])
    trajectory['fuel'] = np.array(trajectory['fuel'])
    trajectory['actions'] = np.array(trajectory['actions']) if trajectory['actions'] else np.array([])
    trajectory['rewards'] = np.array(trajectory['rewards'])
    trajectory['cumulative_rewards'] = np.array(trajectory['cumulative_rewards'])

    return trajectory


def plot_individual_trajectories(trajectories: List[Dict], env, ax):
    """Plot individual rocket trajectories"""
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Individual Trajectories ({len(trajectories)} runs)')
    ax.grid(True, alpha=0.2)

    # Draw environment objects (faint)
    for planet in env.landable_planets:
        circle = Circle((planet.x, planet.y), planet.radius * 2,
                       color='green', alpha=0.2)
        ax.add_patch(circle)

    for bh in env.black_holes:
        circle = Circle((bh.x, bh.y), bh.event_horizon,
                       color='purple', alpha=0.15)
        ax.add_patch(circle)

    for sun in env.suns:
        circle = Circle((sun.x, sun.y), sun.radius * 2,
                       color='yellow', alpha=0.2)
        ax.add_patch(circle)

    # Earth
    earth = Circle(env.earth_pos, 8, color='blue', alpha=0.3, zorder=5)
    ax.add_patch(earth)

    # Plot trajectories with color based on success
    for i, traj in enumerate(trajectories):
        positions = traj['positions']

        if traj['success']:
            color = 'green'
            alpha = 0.6
            linewidth = 1.5
            label = 'Success' if i == 0 else None
        elif traj['termination_reason'] == 'collision':
            color = 'red'
            alpha = 0.4
            linewidth = 1.0
            label = 'Collision' if i == 0 and not any(t['success'] for t in trajectories[:i]) else None
        else:
            color = 'orange'
            alpha = 0.3
            linewidth = 0.8
            label = 'Incomplete' if i == 0 and not any(t['success'] or t['termination_reason'] == 'collision' for t in trajectories[:i]) else None

        ax.plot(positions[:, 0], positions[:, 1],
               color=color, alpha=alpha, linewidth=linewidth, label=label)

        # Mark start and end
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=4, alpha=0.5)
        ax.plot(positions[-1, 0], positions[-1, 1], 'rx', markersize=5, alpha=0.5)

    ax.legend(loc='upper right')


def plot_average_trajectory(trajectories: List[Dict], env, ax):
    """Plot average trajectory with confidence bands"""
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Average Trajectory (with std deviation)')
    ax.grid(True, alpha=0.2)

    # Draw environment objects (faint)
    for planet in env.landable_planets:
        circle = Circle((planet.x, planet.y), planet.radius * 2,
                       color='green', alpha=0.2)
        ax.add_patch(circle)

    for bh in env.black_holes:
        circle = Circle((bh.x, bh.y), bh.event_horizon,
                       color='purple', alpha=0.15)
        ax.add_patch(circle)

    for sun in env.suns:
        circle = Circle((sun.x, sun.y), sun.radius * 2,
                       color='yellow', alpha=0.2)
        ax.add_patch(circle)

    # Earth
    earth = Circle(env.earth_pos, 8, color='blue', alpha=0.3, zorder=5)
    ax.add_patch(earth)

    # Find max trajectory length
    max_length = max(len(traj['positions']) for traj in trajectories)

    # Interpolate all trajectories to same length
    interpolated_positions = []
    for traj in trajectories:
        positions = traj['positions']
        if len(positions) < max_length:
            # Interpolate to max_length
            t_old = np.linspace(0, 1, len(positions))
            t_new = np.linspace(0, 1, max_length)
            x_interp = np.interp(t_new, t_old, positions[:, 0])
            y_interp = np.interp(t_new, t_old, positions[:, 1])
            interpolated_positions.append(np.column_stack([x_interp, y_interp]))
        else:
            interpolated_positions.append(positions)

    # Compute mean and std
    positions_array = np.array(interpolated_positions)  # Shape: (n_trajectories, max_length, 2)
    mean_positions = np.mean(positions_array, axis=0)
    std_positions = np.std(positions_array, axis=0)

    # Plot confidence ellipses at regular intervals
    step_size = max(1, max_length // 20)
    for i in range(0, max_length, step_size):
        ellipse = plt.Circle(mean_positions[i], np.mean(std_positions[i]),
                            color='blue', alpha=0.1, zorder=1)
        ax.add_patch(ellipse)

    # Plot mean trajectory
    ax.plot(mean_positions[:, 0], mean_positions[:, 1],
           'b-', linewidth=3, label='Mean trajectory', zorder=3)

    # Plot confidence bands
    ax.fill_between(mean_positions[:, 0],
                    mean_positions[:, 1] - std_positions[:, 1],
                    mean_positions[:, 1] + std_positions[:, 1],
                    alpha=0.2, color='blue', label='±1 std deviation')

    # Mark start and end
    ax.plot(mean_positions[0, 0], mean_positions[0, 1], 'go',
           markersize=10, label='Start', zorder=4)
    ax.plot(mean_positions[-1, 0], mean_positions[-1, 1], 'r*',
           markersize=15, label='Avg end point', zorder=4)

    ax.legend(loc='upper right')


def plot_heatmap(trajectories: List[Dict], env, ax):
    """Plot heatmap of visited positions"""
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Position Heatmap (visitation frequency)')

    # Collect all positions
    all_positions = []
    for traj in trajectories:
        all_positions.extend(traj['positions'])
    all_positions = np.array(all_positions)

    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        all_positions[:, 0], all_positions[:, 1],
        bins=50, range=[[0, env.grid_size], [0, env.grid_size]]
    )

    # Plot heatmap
    im = ax.imshow(heatmap.T, origin='lower', cmap='hot', alpha=0.7,
                   extent=[0, env.grid_size, 0, env.grid_size],
                   interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Visitation count')

    # Draw environment objects on top
    for planet in env.landable_planets:
        circle = Circle((planet.x, planet.y), planet.radius * 2,
                       color='green', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)

    for bh in env.black_holes:
        circle = Circle((bh.x, bh.y), bh.event_horizon,
                       color='cyan', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)

    # Earth
    ax.plot(env.earth_pos[0], env.earth_pos[1], 'w*', markersize=15,
           markeredgecolor='black', markeredgewidth=1)


def plot_time_series(trajectories: List[Dict], axs):
    """Plot time series data (position, velocity, fuel, rewards)"""

    # 1. Position over time
    ax = axs[0]
    for i, traj in enumerate(trajectories):
        steps = np.arange(len(traj['positions']))
        x_pos = traj['positions'][:, 0]
        y_pos = traj['positions'][:, 1]

        alpha = 0.3 if len(trajectories) > 5 else 0.6
        ax.plot(steps, x_pos, 'b-', alpha=alpha, linewidth=1)
        ax.plot(steps, y_pos, 'r-', alpha=alpha, linewidth=1)

    # Add mean lines
    if len(trajectories) > 1:
        max_len = max(len(t['positions']) for t in trajectories)
        mean_x = []
        mean_y = []
        for step in range(max_len):
            x_vals = [t['positions'][min(step, len(t['positions'])-1), 0] for t in trajectories]
            y_vals = [t['positions'][min(step, len(t['positions'])-1), 1] for t in trajectories]
            mean_x.append(np.mean(x_vals))
            mean_y.append(np.mean(y_vals))

        ax.plot(range(max_len), mean_x, 'b-', linewidth=2.5, label='Mean X')
        ax.plot(range(max_len), mean_y, 'r-', linewidth=2.5, label='Mean Y')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Position')
    ax.set_title('Position over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Speed over time
    ax = axs[1]
    for i, traj in enumerate(trajectories):
        steps = np.arange(len(traj['speeds']))
        alpha = 0.3 if len(trajectories) > 5 else 0.6
        ax.plot(steps, traj['speeds'], alpha=alpha, linewidth=1)

    # Add mean line
    if len(trajectories) > 1:
        max_len = max(len(t['speeds']) for t in trajectories)
        mean_speed = []
        for step in range(max_len):
            speeds = [t['speeds'][min(step, len(t['speeds'])-1)] for t in trajectories]
            mean_speed.append(np.mean(speeds))
        ax.plot(range(max_len), mean_speed, 'k-', linewidth=2.5, label='Mean speed')
        ax.legend()

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed')
    ax.set_title('Speed over Time')
    ax.grid(True, alpha=0.3)

    # 3. Fuel over time
    ax = axs[2]
    for i, traj in enumerate(trajectories):
        steps = np.arange(len(traj['fuel']))
        alpha = 0.3 if len(trajectories) > 5 else 0.6
        color = 'green' if traj['success'] else 'red' if traj['termination_reason'] == 'collision' else 'orange'
        ax.plot(steps, traj['fuel'], color=color, alpha=alpha, linewidth=1)

    # Add mean line
    if len(trajectories) > 1:
        max_len = max(len(t['fuel']) for t in trajectories)
        mean_fuel = []
        for step in range(max_len):
            fuels = [t['fuel'][min(step, len(t['fuel'])-1)] for t in trajectories]
            mean_fuel.append(np.mean(fuels))
        ax.plot(range(max_len), mean_fuel, 'k-', linewidth=2.5, label='Mean fuel')
        ax.legend()

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Fuel Remaining')
    ax.set_title('Fuel Consumption over Time')
    ax.grid(True, alpha=0.3)

    # 4. Cumulative reward over time
    ax = axs[3]
    for i, traj in enumerate(trajectories):
        steps = np.arange(len(traj['cumulative_rewards']))
        alpha = 0.3 if len(trajectories) > 5 else 0.6
        color = 'green' if traj['success'] else 'red' if traj['termination_reason'] == 'collision' else 'orange'
        ax.plot(steps, traj['cumulative_rewards'], color=color, alpha=alpha, linewidth=1)

    # Add mean line
    if len(trajectories) > 1:
        max_len = max(len(t['cumulative_rewards']) for t in trajectories)
        mean_reward = []
        for step in range(max_len):
            rewards = [t['cumulative_rewards'][min(step, len(t['cumulative_rewards'])-1)] for t in trajectories]
            mean_reward.append(np.mean(rewards))
        ax.plot(range(max_len), mean_reward, 'k-', linewidth=2.5, label='Mean cumulative reward')
        ax.legend()

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward over Time')
    ax.grid(True, alpha=0.3)


def plot_statistics(trajectories: List[Dict], ax):
    """Plot summary statistics"""
    ax.axis('off')
    ax.set_title('Run Statistics', fontsize=14, fontweight='bold')

    # Compute statistics
    n_runs = len(trajectories)
    n_success = sum(t['success'] for t in trajectories)
    n_collision = sum(t['termination_reason'] == 'collision' for t in trajectories)
    n_other = n_runs - n_success - n_collision

    avg_steps = np.mean([t['steps'] for t in trajectories])
    avg_reward = np.mean([t['cumulative_rewards'][-1] for t in trajectories])
    avg_final_fuel = np.mean([t['fuel'][-1] for t in trajectories])

    # Distances
    distances = []
    for traj in trajectories:
        final_pos = traj['positions'][-1]
        distances.append(np.linalg.norm(final_pos - trajectories[0]['positions'][0]))
    avg_distance = np.mean(distances)

    stats_text = f"""
RUN SUMMARY
{'='*40}

Total Runs: {n_runs}

Outcomes:
  • Successful Landings: {n_success} ({n_success/n_runs*100:.1f}%)
  • Collisions: {n_collision} ({n_collision/n_runs*100:.1f}%)
  • Other/Timeout: {n_other} ({n_other/n_runs*100:.1f}%)

Performance Metrics:
  • Avg Steps: {avg_steps:.1f}
  • Avg Cumulative Reward: {avg_reward:.2f}
  • Avg Final Fuel: {avg_final_fuel:.1f}
  • Avg Distance from Start: {avg_distance:.1f}

Per-Run Details:
"""

    for i, traj in enumerate(trajectories):
        outcome = '✓ Landing' if traj['success'] else '✗ Collision' if traj['termination_reason'] == 'collision' else '○ Timeout'
        stats_text += f"  Run {i+1}: {outcome} | "
        stats_text += f"Steps: {traj['steps']} | "
        stats_text += f"Reward: {traj['cumulative_rewards'][-1]:.1f}\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))


def main():
    parser = argparse.ArgumentParser(description='Visualize Agent Performance with Trajectories')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=['qlearning', 'dqn', 'ppo'],
                       help='Algorithm type')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: models/{algorithm}_agent_final.[pkl|pt])')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs to visualize (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Environment seed (default: 42)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save visualization to file')

    args = parser.parse_args()

    print("="*60)
    print("ROCKET RUN VISUALIZATION")
    print("="*60)

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        ext = 'pkl' if args.algorithm == 'qlearning' else 'pt'
        model_path = f"models/{args.algorithm}_agent_final.{ext}"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train a model first with: python train.py --algorithm {args.algorithm} --episodes 500")
        return

    # Load agent
    print(f"\nLoading {args.algorithm.upper()} agent from {model_path}...")
    agent = load_agent(args.algorithm, model_path)

    # Create environment
    print(f"Creating environment with seed {args.seed}...")
    env = AstrophysicsEnv(
        grid_size=1000,
        max_fuel=500.0,
        max_steps=args.max_steps,
        seed=args.seed,
        static_environment=True
    )
    env.reset()

    # Run episodes
    print(f"\nRunning {args.runs} episodes...")
    trajectories = []
    for i in range(args.runs):
        print(f"  Run {i+1}/{args.runs}...", end=' ')
        traj = run_episode(env, agent, args.algorithm, args.max_steps)
        trajectories.append(traj)
        outcome = 'Success' if traj['success'] else 'Collision' if traj['termination_reason'] == 'collision' else 'Timeout'
        print(f"{outcome} (steps: {traj['steps']}, reward: {traj['cumulative_rewards'][-1]:.1f})")

    print("\nGenerating visualizations...")

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Top row: spatial visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    plot_individual_trajectories(trajectories, env, ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_average_trajectory(trajectories, env, ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_heatmap(trajectories, env, ax3)

    ax_stats = fig.add_subplot(gs[0, 3])
    plot_statistics(trajectories, ax_stats)

    # Middle and bottom rows: time series
    time_series_axes = [
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4])
    ]
    plot_time_series(trajectories, time_series_axes)

    plt.suptitle(f'{args.algorithm.upper()} Agent Performance - {args.runs} Runs (Seed: {args.seed})',
                fontsize=16, fontweight='bold')

    if args.save:
        print(f"\nSaving to {args.save}...")
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print("Saved!")

    print("\nDisplaying visualization...")
    print("Close the window to exit.")
    plt.show()

    env.close()


if __name__ == "__main__":
    main()
