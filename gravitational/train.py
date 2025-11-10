"""
Training script for Gravitational Dynamics RL Environment
Includes convergence tracking and visualization.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gravitational_env import GravitationalDynamicsEnv
from dqn_agent import DQNAgent, RandomAgent, HeuristicAgent
import json
import os
from datetime import datetime
from typing import Dict, List


class ConvergenceTracker:
    """Track and analyze training convergence"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = []
        self.black_hole_count = []
        self.losses = []
        self.epsilons = []

    def add_episode(self, reward: float, length: int, success: bool, black_hole: bool,
                   loss: float = 0.0, epsilon: float = 0.0):
        """Add episode statistics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_count.append(1 if success else 0)
        self.black_hole_count.append(1 if black_hole else 0)
        self.losses.append(loss)
        self.epsilons.append(epsilon)

    def get_moving_average(self, data: List[float], window: int = None) -> List[float]:
        """Calculate moving average"""
        if window is None:
            window = self.window_size

        if len(data) < window:
            return [np.mean(data[:i+1]) for i in range(len(data))]

        moving_avg = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(data[start:i+1]))
        return moving_avg

    def get_success_rate(self, window: int = None) -> List[float]:
        """Calculate success rate over moving window"""
        return self.get_moving_average(self.success_count, window)

    def is_converged(self, threshold: float = 0.7, window: int = 100,
                    min_episodes: int = 500) -> bool:
        """
        Check if training has converged.

        Criteria:
        - Success rate > threshold over last 'window' episodes
        - At least 'min_episodes' completed
        """
        if len(self.episode_rewards) < min_episodes:
            return False

        recent_success_rate = np.mean(self.success_count[-window:])
        return recent_success_rate >= threshold

    def get_statistics(self) -> Dict:
        """Get current training statistics"""
        if len(self.episode_rewards) == 0:
            return {}

        window = min(self.window_size, len(self.episode_rewards))

        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_reward_recent': np.mean(self.episode_rewards[-window:]),
            'avg_length': np.mean(self.episode_lengths),
            'avg_length_recent': np.mean(self.episode_lengths[-window:]),
            'success_rate': np.mean(self.success_count),
            'success_rate_recent': np.mean(self.success_count[-window:]),
            'black_hole_rate': np.mean(self.black_hole_count),
            'black_hole_rate_recent': np.mean(self.black_hole_count[-window:]),
            'current_epsilon': self.epsilons[-1] if self.epsilons else 0.0
        }

    def plot_training_progress(self, save_path: str = "training_progress.png"):
        """Plot training progress with multiple metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        episodes = list(range(1, len(self.episode_rewards) + 1))

        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(episodes, self.episode_rewards, alpha=0.3, label='Episode Reward')
        ax1.plot(episodes, self.get_moving_average(self.episode_rewards),
                linewidth=2, label=f'{self.window_size}-Episode Moving Avg')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Episode Rewards Over Time')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Success Rate
        ax2 = axes[0, 1]
        success_rate = self.get_success_rate()
        ax2.plot(episodes, success_rate, linewidth=2, color='green')
        ax2.axhline(y=0.7, color='r', linestyle='--', label='Target (70%)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title(f'Success Rate (Moving Window: {self.window_size})')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Plot 3: Episode Length
        ax3 = axes[0, 2]
        ax3.plot(episodes, self.episode_lengths, alpha=0.3, label='Episode Length')
        ax3.plot(episodes, self.get_moving_average(self.episode_lengths),
                linewidth=2, label=f'{self.window_size}-Episode Moving Avg')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.set_title('Episode Length Over Time')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Plot 4: Black Hole Rate
        ax4 = axes[1, 0]
        black_hole_rate = self.get_moving_average(self.black_hole_count)
        ax4.plot(episodes, black_hole_rate, linewidth=2, color='red')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Black Hole Rate')
        ax4.set_title(f'Black Hole Death Rate (Moving Window: {self.window_size})')
        ax4.set_ylim([0, 1])
        ax4.grid(alpha=0.3)

        # Plot 5: Loss
        ax5 = axes[1, 1]
        if len(self.losses) > 0 and max(self.losses) > 0:
            ax5.plot(episodes, self.losses, alpha=0.3, label='Loss')
            ax5.plot(episodes, self.get_moving_average(self.losses),
                    linewidth=2, label=f'{self.window_size}-Episode Moving Avg')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Loss')
            ax5.set_title('Training Loss Over Time')
            ax5.legend()
            ax5.grid(alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No loss data available', ha='center', va='center',
                    transform=ax5.transAxes)
            ax5.set_title('Training Loss')

        # Plot 6: Epsilon
        ax6 = axes[1, 2]
        if len(self.epsilons) > 0:
            ax6.plot(episodes, self.epsilons, linewidth=2, color='purple')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Epsilon')
            ax6.set_title('Exploration Rate (Epsilon) Over Time')
            ax6.set_ylim([0, 1])
            ax6.grid(alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No epsilon data available', ha='center', va='center',
                    transform=ax6.transAxes)
            ax6.set_title('Exploration Rate')

        plt.suptitle('Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to: {save_path}")
        plt.close()


def train_agent(
    agent,
    env: GravitationalDynamicsEnv,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    save_dir: str = "checkpoints",
    save_freq: int = 100,
    eval_freq: int = 50,
    convergence_threshold: float = 0.7,
    verbose: bool = True
):
    """
    Train an RL agent on the gravitational environment.

    Args:
        agent: RL agent with select_action, store_transition, update, decay_epsilon methods
        env: GravitationalDynamicsEnv instance
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        save_dir: Directory to save checkpoints and logs
        save_freq: Save checkpoint every N episodes
        eval_freq: Print evaluation every N episodes
        convergence_threshold: Success rate threshold for convergence
        verbose: Print training progress

    Returns:
        tracker: ConvergenceTracker with training history
    """
    os.makedirs(save_dir, exist_ok=True)

    tracker = ConvergenceTracker(window_size=100)

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        success = False
        black_hole_death = False

        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward, next_state, terminated)

            # Update agent
            loss = agent.update()
            if loss > 0:
                episode_loss += loss
                loss_count += 1

            episode_reward += reward
            state = next_state

            # Check terminal conditions
            if terminated:
                # Check if success or black hole death
                if reward == 0:  # Target planet reward
                    success = True
                elif reward == -100:  # Black hole reward
                    black_hole_death = True
                break

            if truncated:
                break

        # Decay exploration
        agent.decay_epsilon()

        # Track statistics
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        epsilon = agent.epsilon if hasattr(agent, 'epsilon') else 0.0
        tracker.add_episode(
            reward=episode_reward,
            length=step + 1,
            success=success,
            black_hole=black_hole_death,
            loss=avg_loss,
            epsilon=epsilon
        )

        # Print progress
        if verbose and episode % eval_freq == 0:
            stats = tracker.get_statistics()
            print(f"\n{'='*80}")
            print(f"Episode {episode}/{n_episodes}")
            print(f"{'='*80}")
            print(f"  Avg Reward (recent):       {stats['avg_reward_recent']:>8.2f}")
            print(f"  Avg Length (recent):       {stats['avg_length_recent']:>8.2f}")
            print(f"  Success Rate (recent):     {stats['success_rate_recent']:>8.2%}")
            print(f"  Black Hole Rate (recent):  {stats['black_hole_rate_recent']:>8.2%}")
            print(f"  Current Epsilon:           {stats['current_epsilon']:>8.4f}")
            print(f"  Current Episode Reward:    {episode_reward:>8.2f}")
            print(f"  Current Episode Length:    {step + 1:>8}")
            print(f"  Success: {success}, Black Hole: {black_hole_death}")

        # Save checkpoint
        if hasattr(agent, 'save') and episode % save_freq == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_ep{episode}.pt")
            agent.save(checkpoint_path)
            if verbose:
                print(f"  Checkpoint saved: {checkpoint_path}")

        # Check convergence
        if tracker.is_converged(threshold=convergence_threshold, window=100, min_episodes=500):
            if verbose:
                print(f"\n{'='*80}")
                print(f"CONVERGED at episode {episode}!")
                print(f"Success rate: {tracker.get_success_rate()[-1]:.2%}")
                print(f"{'='*80}")
            break

    # Final save
    if hasattr(agent, 'save'):
        final_path = os.path.join(save_dir, "final_model.pt")
        agent.save(final_path)
        print(f"\nFinal model saved: {final_path}")

    # Save statistics
    stats = tracker.get_statistics()
    stats_path = os.path.join(save_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Training statistics saved: {stats_path}")

    # Plot training progress
    plot_path = os.path.join(save_dir, "training_progress.png")
    tracker.plot_training_progress(plot_path)

    return tracker


if __name__ == "__main__":
    print("="*80)
    print("Training RL Agent on Gravitational Dynamics Environment")
    print("="*80)

    # Create environment
    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=0.1)

    # Select agent type
    agent_type = "dqn"  # Options: "dqn", "random", "heuristic"

    if agent_type == "dqn":
        print("\nAgent: Deep Q-Network (DQN)")
        agent = DQNAgent(
            state_dim=4,
            n_directions=4,
            n_thrust_levels=5,
            hidden_dims=[128, 128],
            learning_rate=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_capacity=10000,
            batch_size=64
        )
        save_dir = "checkpoints/dqn"
    elif agent_type == "random":
        print("\nAgent: Random Baseline")
        agent = RandomAgent()
        save_dir = "checkpoints/random"
    elif agent_type == "heuristic":
        print("\nAgent: Heuristic")
        agent = HeuristicAgent(env)
        save_dir = "checkpoints/heuristic"
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Training parameters
    n_episodes = 2000
    max_steps = 1000

    print(f"\nTraining Configuration:")
    print(f"  Episodes: {n_episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Save directory: {save_dir}")
    print(f"\nStarting training...\n")

    # Train
    tracker = train_agent(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        max_steps=max_steps,
        save_dir=save_dir,
        save_freq=100,
        eval_freq=50,
        convergence_threshold=0.7,
        verbose=True
    )

    # Final statistics
    stats = tracker.get_statistics()
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Total Episodes:           {stats['total_episodes']}")
    print(f"Average Reward:           {stats['avg_reward']:.2f}")
    print(f"Average Length:           {stats['avg_length']:.2f}")
    print(f"Overall Success Rate:     {stats['success_rate']:.2%}")
    print(f"Recent Success Rate:      {stats['success_rate_recent']:.2%}")
    print(f"Overall Black Hole Rate:  {stats['black_hole_rate']:.2%}")
    print(f"Recent Black Hole Rate:   {stats['black_hole_rate_recent']:.2%}")
    print(f"{'='*80}")
