"""
Main training script for all three RL algorithms
"""

import argparse
import os
import numpy as np
from astrophysics_env import AstrophysicsEnv
from q_learning_agent import DiscretizedQLearningAgent, train_q_learning
from dqn_agent import DQNAgent, train_dqn
from ppo_agent import PPOAgent, train_ppo


def create_directories():
    """Create necessary directories"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)


def train_agent(algorithm: str, num_episodes: int = 500, seed: int = 42, dynamic: bool = False):
    """
    Train an agent using the specified algorithm

    Args:
        algorithm: One of 'qlearning', 'dqn', 'ppo'
        num_episodes: Number of training episodes
        seed: Random seed
        dynamic: If True, use dynamic environment (celestial objects change each episode)
                If False (default), use static environment (same layout each episode)
    """
    create_directories()

    print(f"\n{'='*60}")
    print(f"Training {algorithm.upper()} Agent")
    print(f"{'='*60}\n")

    # Create environment
    # static_environment=True (default): Same layout each episode - easier to learn
    # static_environment=False: Random layout each episode - more generalizable
    env = AstrophysicsEnv(
        grid_size=1000,
        max_fuel=500.0,
        max_steps=500,
        seed=seed,
        static_environment=not dynamic
    )

    env_type = "Dynamic" if dynamic else "Static"
    print(f"Environment Mode: {env_type}")

    if algorithm.lower() == 'qlearning':
        # Q-Learning with discretization
        agent = DiscretizedQLearningAgent(
            state_bins=(20, 20, 10, 10, 10, 10, 8, 10, 8, 10, 8),
            action_bins=(5, 8),
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )

        trained_agent = train_q_learning(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            max_steps_per_episode=500,
            save_path="models/q_learning_agent"
        )

    elif algorithm.lower() == 'dqn':
        # Deep Q-Network
        agent = DQNAgent(
            state_dim=11,
            action_bins=(7, 12),
            learning_rate=0.001,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            buffer_capacity=50000,
            batch_size=64,
            target_update_freq=10
        )

        trained_agent = train_dqn(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            max_steps_per_episode=500,
            update_freq=4,
            save_path="models/dqn_agent"
        )

    elif algorithm.lower() == 'ppo':
        # Proximal Policy Optimization
        agent = PPOAgent(
            state_dim=11,
            action_dim=2,
            learning_rate=3e-4,
            discount_factor=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            num_epochs=10,
            batch_size=64
        )

        trained_agent = train_ppo(
            env=env,
            agent=agent,
            num_episodes=num_episodes,
            max_steps_per_episode=500,
            update_interval=2048,
            save_path="models/ppo_agent"
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'qlearning', 'dqn', 'ppo'")

    # Save training statistics
    import pickle
    stats_path = f"results/{algorithm}_training_stats.pkl"
    with open(stats_path, 'wb') as f:
        pickle.dump(trained_agent.training_stats, f)
    print(f"\nSaved training statistics to {stats_path}")

    env.close()
    return trained_agent


def plot_training_results(algorithm: str):
    """
    Plot training results

    Args:
        algorithm: Algorithm name
    """
    import pickle
    import matplotlib.pyplot as plt

    stats_path = f"results/{algorithm}_training_stats.pkl"
    if not os.path.exists(stats_path):
        print(f"Statistics file not found: {stats_path}")
        return

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{algorithm.upper()} Training Results', fontsize=16)

    # Episode rewards
    axes[0, 0].plot(stats['episode_rewards'])
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True)

    # Moving average of rewards
    window = 50
    if len(stats['episode_rewards']) >= window:
        moving_avg = np.convolve(stats['episode_rewards'],
                                np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_title(f'Moving Average Reward (window={window})')
        axes[0, 1].grid(True)

    # Episode lengths
    axes[1, 0].plot(stats['episode_lengths'])
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].grid(True)

    # Algorithm-specific plots
    if algorithm.lower() in ['qlearning', 'dqn'] and 'epsilon_values' in stats:
        axes[1, 1].plot(stats['epsilon_values'])
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].grid(True)
    elif algorithm.lower() == 'ppo' and 'policy_losses' in stats:
        axes[1, 1].plot(stats['policy_losses'], label='Policy Loss')
        if 'value_losses' in stats:
            axes[1, 1].plot(stats['value_losses'], label='Value Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()
    plot_path = f"results/{algorithm}_training_plot.png"
    plt.savefig(plot_path)
    print(f"Saved training plot to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train RL agents for Astrophysics Environment')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=['qlearning', 'dqn', 'ppo', 'all'],
                       help='Algorithm to train (qlearning, dqn, ppo, or all)')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--dynamic', action='store_true',
                       help='Use dynamic environment (celestial objects change each episode). '
                            'Default is static environment (same layout each episode).')
    parser.add_argument('--plot', action='store_true',
                       help='Plot training results after training')

    args = parser.parse_args()

    if args.algorithm == 'all':
        algorithms = ['qlearning', 'dqn', 'ppo']
    else:
        algorithms = [args.algorithm]

    for algo in algorithms:
        train_agent(algo, args.episodes, args.seed, args.dynamic)
        if args.plot:
            plot_training_results(algo)

    print("\n" + "="*60)
    print("Training completed for all algorithms!")
    print("="*60)


if __name__ == "__main__":
    main()
