"""
Evaluation script for trained RL agents
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from astrophysics_env import AstrophysicsEnv
from q_learning_agent import DiscretizedQLearningAgent
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent


def load_agent(algorithm: str, model_path: str):
    """
    Load a trained agent

    Args:
        algorithm: Algorithm name
        model_path: Path to model file

    Returns:
        Loaded agent
    """
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


def evaluate_agent(env, agent, algorithm: str, num_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained agent

    Args:
        env: Environment
        agent: Trained agent
        algorithm: Algorithm name
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {algorithm.upper()} agent for {num_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []
    success_count = 0  # Count of successful landings

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get action (no exploration during evaluation)
            if algorithm.lower() == 'qlearning':
                action = agent.get_action(state, training=False)
            elif algorithm.lower() == 'dqn':
                action, _ = agent.get_action(state, training=False)
            elif algorithm.lower() == 'ppo':
                action, _, _ = agent.get_action(state, training=False)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            state = next_state

            if render:
                env.render()

            # Check if successfully landed (reward = 0 means landing on valid planet)
            if terminated and reward == 0:
                success_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}, "
              f"Final Pos = ({info['rocket_pos'][0]:.1f}, {info['rocket_pos'][1]:.1f})")

    # Compute statistics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

    print(f"\n{'='*60}")
    print(f"Evaluation Results for {algorithm.upper()}")
    print(f"{'='*60}")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")

    return metrics


def run_single_episode_with_visualization(env, agent, algorithm: str, save_path: str = None):
    """
    Run a single episode with detailed visualization

    Args:
        env: Environment
        agent: Trained agent
        algorithm: Algorithm name
        save_path: Path to save trajectory plot
    """
    print(f"\nRunning visualization episode for {algorithm.upper()}...")

    state, info = env.reset()
    trajectory = [info['rocket_pos'].copy()]
    velocities = [info['rocket_vel'].copy()]
    fuels = [info['fuel']]
    actions_taken = []
    rewards_received = []

    done = False
    total_reward = 0

    while not done:
        # Get action
        if algorithm.lower() == 'qlearning':
            action = agent.get_action(state, training=False)
        elif algorithm.lower() == 'dqn':
            action, _ = agent.get_action(state, training=False)
        elif algorithm.lower() == 'ppo':
            action, _, _ = agent.get_action(state, training=False)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Record data
        trajectory.append(info['rocket_pos'].copy())
        velocities.append(info['rocket_vel'].copy())
        fuels.append(info['fuel'])
        actions_taken.append(action.copy())
        rewards_received.append(reward)
        total_reward += reward

        state = next_state

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{algorithm.upper()} - Episode Trajectory (Total Reward: {total_reward:.2f})',
                 fontsize=16)

    # Plot 1: Trajectory in 2D space
    trajectory = np.array(trajectory)
    axes[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Rocket Path')
    axes[0, 0].scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=200,
                      marker='o', label='Start (Earth)', zorder=5)
    axes[0, 0].scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=200,
                      marker='X', label='End', zorder=5)

    # Plot celestial objects
    for planet in env.landable_planets:
        circle = plt.Circle((planet.x, planet.y), planet.radius*3, color='green', alpha=0.3)
        axes[0, 0].add_patch(circle)
    for bh in env.black_holes:
        circle = plt.Circle((bh.x, bh.y), bh.event_horizon, color='purple', alpha=0.3)
        axes[0, 0].add_patch(circle)
    for sun in env.suns:
        circle = plt.Circle((sun.x, sun.y), sun.radius*2, color='yellow', alpha=0.5)
        axes[0, 0].add_patch(circle)

    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].set_title('Rocket Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, env.grid_size)
    axes[0, 0].set_ylim(0, env.grid_size)

    # Plot 2: Velocity over time
    velocities = np.array(velocities)
    speeds = np.linalg.norm(velocities, axis=1)
    axes[0, 1].plot(speeds, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Speed')
    axes[0, 1].set_title('Rocket Speed Over Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Fuel consumption
    axes[1, 0].plot(fuels, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Remaining Fuel')
    axes[1, 0].set_title('Fuel Consumption')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Rewards over time
    cumulative_rewards = np.cumsum(rewards_received)
    axes[1, 1].plot(rewards_received, 'orange', linewidth=1, alpha=0.5, label='Instant Reward')
    axes[1, 1].plot(cumulative_rewards, 'b-', linewidth=2, label='Cumulative Reward')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Rewards Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved trajectory visualization to {save_path}")

    plt.show()


def compare_algorithms(metrics_dict: dict, save_path: str = None):
    """
    Compare multiple algorithms

    Args:
        metrics_dict: Dictionary of {algorithm_name: metrics}
        save_path: Path to save comparison plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Algorithm Comparison', fontsize=16)

    algorithms = list(metrics_dict.keys())
    colors = ['blue', 'orange', 'green']

    # Mean rewards
    mean_rewards = [metrics_dict[algo]['mean_reward'] for algo in algorithms]
    std_rewards = [metrics_dict[algo]['std_reward'] for algo in algorithms]
    axes[0].bar(algorithms, mean_rewards, yerr=std_rewards, color=colors, alpha=0.7)
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Mean Episode Reward')
    axes[0].grid(True, alpha=0.3)

    # Mean lengths
    mean_lengths = [metrics_dict[algo]['mean_length'] for algo in algorithms]
    std_lengths = [metrics_dict[algo]['std_length'] for algo in algorithms]
    axes[1].bar(algorithms, mean_lengths, yerr=std_lengths, color=colors, alpha=0.7)
    axes[1].set_ylabel('Mean Episode Length')
    axes[1].set_title('Mean Episode Length')
    axes[1].grid(True, alpha=0.3)

    # Success rates
    success_rates = [metrics_dict[algo]['success_rate'] * 100 for algo in algorithms]
    axes[2].bar(algorithms, success_rates, color=colors, alpha=0.7)
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_title('Landing Success Rate')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL agents')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=['qlearning', 'dqn', 'ppo', 'all'],
                       help='Algorithm to evaluate')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (if not provided, uses default path)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create detailed trajectory visualization')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed for evaluation (default: 123)')

    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Create environment
    render_mode = 'human' if args.render else None
    env = AstrophysicsEnv(
        grid_size=1000,
        max_fuel=500.0,
        max_steps=500,
        render_mode=render_mode,
        seed=args.seed
    )

    if args.algorithm == 'all':
        algorithms = ['qlearning', 'dqn', 'ppo']
        metrics_dict = {}

        for algo in algorithms:
            model_path = f"models/{algo}_agent_final.pkl" if algo == 'qlearning' else f"models/{algo}_agent_final.pt"
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}. Skipping {algo}.")
                continue

            print(f"\n{'='*60}")
            print(f"Loading {algo.upper()} agent from {model_path}")
            print(f"{'='*60}")

            agent = load_agent(algo, model_path)
            metrics = evaluate_agent(env, agent, algo, args.episodes, args.render)
            metrics_dict[algo] = metrics

            if args.visualize:
                run_single_episode_with_visualization(
                    env, agent, algo,
                    save_path=f"results/{algo}_trajectory.png"
                )

        # Compare algorithms
        if len(metrics_dict) > 1:
            compare_algorithms(metrics_dict, save_path="results/algorithm_comparison.png")

    else:
        # Single algorithm
        if args.model:
            model_path = args.model
        else:
            ext = 'pkl' if args.algorithm == 'qlearning' else 'pt'
            model_path = f"models/{args.algorithm}_agent_final.{ext}"

        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            return

        print(f"Loading {args.algorithm.upper()} agent from {model_path}")
        agent = load_agent(args.algorithm, model_path)
        evaluate_agent(env, agent, args.algorithm, args.episodes, args.render)

        if args.visualize:
            run_single_episode_with_visualization(
                env, agent, args.algorithm,
                save_path=f"results/{args.algorithm}_trajectory.png"
            )

    env.close()


if __name__ == "__main__":
    main()
