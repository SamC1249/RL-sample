"""
Visualize PPO Policy Actions and Trajectories
Load trained model and see what actions it takes in the environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
from gravitational_env import GravitationalDynamicsEnv
from pathlib import Path
import argparse


class PolicyVisualizer:
    """Visualize trained policy behavior in the environment"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to trained model (.zip file)
        """
        print(f"Loading model from: {model_path}")
        self.model = PPO.load(model_path)
        self.env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=1.0)
        
    def rollout_episode(self, deterministic: bool = True, max_steps: int = 1000):
        """
        Run one episode and record all states, actions, rewards.
        
        Args:
            deterministic: Use deterministic policy (no exploration)
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with trajectory data
        """
        obs, _ = self.env.reset()
        
        trajectory = {
            'states': [obs.copy()],
            'actions': [],
            'rewards': [],
            'positions': [obs[:2].copy()],
            'velocities': [obs[2:].copy()],
            'directions': [],
            'thrusts': [],
            'terminated': False,
            'truncated': False,
            'success': False,
            'black_hole': False,
            'total_reward': 0.0,
            'episode_length': 0
        }
        
        for step in range(max_steps):
            # Get action from policy
            action, _states = self.model.predict(obs, deterministic=deterministic)
            
            # Record action details
            direction = int(action[0])
            thrust = float(action[1])
            trajectory['actions'].append(action.copy())
            trajectory['directions'].append(direction)
            trajectory['thrusts'].append(thrust)
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Record results
            trajectory['states'].append(obs.copy())
            trajectory['positions'].append(obs[:2].copy())
            trajectory['velocities'].append(obs[2:].copy())
            trajectory['rewards'].append(reward)
            trajectory['total_reward'] += reward
            trajectory['episode_length'] += 1
            
            if terminated or truncated:
                trajectory['terminated'] = terminated
                trajectory['truncated'] = truncated
                
                if terminated and info.get('terminal_reason') == 'success':
                    trajectory['success'] = True
                elif terminated and info.get('terminal_reason') == 'black_hole':
                    trajectory['black_hole'] = True
                
                break
        
        return trajectory
    
    def plot_trajectory(self, trajectory: dict, save_path: str = None):
        """
        Plot agent trajectory on the grid with celestial bodies.
        
        Args:
            trajectory: Trajectory data from rollout_episode
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Trajectory on grid
        ax1 = axes[0, 0]
        positions = np.array(trajectory['positions'])
        
        # Plot celestial bodies
        for body in self.env.celestial_bodies:
            if body.type == 'black_hole':
                circle = patches.Circle((body.x, body.y), body.event_horizon, 
                                       color='black', alpha=0.8, label='Black Hole')
                ax1.add_patch(circle)
                ax1.plot(body.x, body.y, 'x', color='red', markersize=15, markeredgewidth=3)
            elif body.type == 'target_planet':
                circle = patches.Circle((body.x, body.y), 2.0, 
                                       color='green', alpha=0.5, label='Target')
                ax1.add_patch(circle)
            else:
                circle = patches.Circle((body.x, body.y), 1.5, 
                                       color='gray', alpha=0.3)
                ax1.add_patch(circle)
        
        # Plot trajectory
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=10, label='End')
        
        # Add velocity arrows (sample every 10 steps)
        for i in range(0, len(positions), 10):
            if i < len(trajectory['velocities']):
                vel = trajectory['velocities'][i]
                vel_norm = np.linalg.norm(vel)
                if vel_norm > 0.1:
                    ax1.arrow(positions[i, 0], positions[i, 1], 
                            vel[0]*2, vel[1]*2,
                            head_width=1.5, head_length=1.0, fc='red', ec='red', alpha=0.5)
        
        ax1.set_xlim([0, 100])
        ax1.set_ylim([0, 100])
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'Agent Trajectory ({trajectory["episode_length"]} steps)\n'
                     f'Success: {trajectory["success"]}, Black Hole: {trajectory["black_hole"]}')
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. Actions over time
        ax2 = axes[0, 1]
        steps = np.arange(len(trajectory['directions']))
        
        # Direction as bar chart
        direction_names = ['Up', 'Right', 'Down', 'Left']
        colors = ['blue', 'green', 'red', 'orange']
        for i, dir_idx in enumerate(trajectory['directions']):
            ax2.bar(i, 1, color=colors[dir_idx], alpha=0.6, width=1.0)
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Direction')
        ax2.set_title('Direction Actions Over Time')
        ax2.set_yticks([0.5, 1.5, 2.5, 3.5])
        ax2.set_yticklabels(direction_names)
        ax2.set_ylim([0, 4])
        ax2.grid(alpha=0.3, axis='x')
        
        # 3. Thrust magnitude over time
        ax3 = axes[1, 0]
        ax3.plot(steps, trajectory['thrusts'], linewidth=2, color='purple', alpha=0.7)
        ax3.fill_between(steps, trajectory['thrusts'], alpha=0.3, color='purple')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Thrust Magnitude')
        ax3.set_title('Thrust Actions Over Time')
        ax3.set_ylim([0, 1])
        ax3.grid(alpha=0.3)
        
        # 4. Rewards over time
        ax4 = axes[1, 1]
        cumulative_reward = np.cumsum(trajectory['rewards'])
        ax4.plot(steps, trajectory['rewards'], linewidth=1, alpha=0.5, 
                color='blue', label='Step Reward')
        ax4.plot(steps, cumulative_reward, linewidth=2, color='darkblue', 
                label='Cumulative Reward')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Reward')
        ax4.set_title(f'Rewards Over Time (Total: {trajectory["total_reward"]:.2f})')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.suptitle('Policy Behavior Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory plot to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_action_distribution(self, n_episodes: int = 100):
        """
        Analyze action distribution across multiple episodes.
        
        Args:
            n_episodes: Number of episodes to analyze
            
        Returns:
            Dictionary with action statistics
        """
        print(f"\nAnalyzing action distribution across {n_episodes} episodes...")
        
        all_directions = []
        all_thrusts = []
        success_count = 0
        black_hole_count = 0
        total_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            trajectory = self.rollout_episode(deterministic=True)
            
            all_directions.extend(trajectory['directions'])
            all_thrusts.extend(trajectory['thrusts'])
            total_rewards.append(trajectory['total_reward'])
            episode_lengths.append(trajectory['episode_length'])
            
            if trajectory['success']:
                success_count += 1
            elif trajectory['black_hole']:
                black_hole_count += 1
            
            if (ep + 1) % 20 == 0:
                print(f"  Completed {ep + 1}/{n_episodes} episodes...")
        
        stats = {
            'direction_counts': {
                'Up': all_directions.count(0),
                'Right': all_directions.count(1),
                'Down': all_directions.count(2),
                'Left': all_directions.count(3)
            },
            'mean_thrust': np.mean(all_thrusts),
            'std_thrust': np.std(all_thrusts),
            'success_rate': success_count / n_episodes,
            'black_hole_rate': black_hole_count / n_episodes,
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths)
        }
        
        return stats
    
    def plot_action_statistics(self, stats: dict, save_path: str = None):
        """
        Plot action distribution statistics.
        
        Args:
            stats: Statistics from analyze_action_distribution
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Direction distribution
        ax1 = axes[0]
        directions = list(stats['direction_counts'].keys())
        counts = list(stats['direction_counts'].values())
        colors = ['blue', 'green', 'red', 'orange']
        
        ax1.bar(directions, counts, color=colors, alpha=0.7)
        ax1.set_xlabel('Direction')
        ax1.set_ylabel('Count')
        ax1.set_title('Action Direction Distribution (100 episodes)')
        ax1.grid(alpha=0.3, axis='y')
        
        # Add percentages
        total = sum(counts)
        for i, (direction, count) in enumerate(zip(directions, counts)):
            percentage = count / total * 100
            ax1.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')
        
        # 2. Performance metrics
        ax2 = axes[1]
        metrics = ['Success\nRate', 'Black Hole\nRate', 'Mean\nReward', 'Mean\nLength']
        values = [
            stats['success_rate'] * 100,
            stats['black_hole_rate'] * 100,
            stats['mean_reward'] / 10,  # Scale for visibility
            stats['mean_episode_length'] / 10  # Scale for visibility
        ]
        colors_metrics = ['green', 'red', 'blue', 'purple']
        
        bars = ax2.bar(metrics, values, color=colors_metrics, alpha=0.7)
        ax2.set_ylabel('Value (scaled)')
        ax2.set_title('Performance Metrics (100 episodes)')
        ax2.grid(alpha=0.3, axis='y')
        
        # Add actual values as text
        ax2.text(0, values[0], f'{stats["success_rate"]:.1%}', ha='center', va='bottom')
        ax2.text(1, values[1], f'{stats["black_hole_rate"]:.1%}', ha='center', va='bottom')
        ax2.text(2, values[2], f'{stats["mean_reward"]:.1f}', ha='center', va='bottom')
        ax2.text(3, values[3], f'{stats["mean_episode_length"]:.0f}', ha='center', va='bottom')
        
        plt.suptitle('Policy Action Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved action statistics to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_action_analysis(self, trajectory: dict):
        """Print detailed action analysis for a single episode"""
        print("\n" + "="*80)
        print("POLICY ACTION ANALYSIS")
        print("="*80)
        
        print(f"\nEpisode Outcome:")
        print(f"  Success: {trajectory['success']}")
        print(f"  Black Hole Death: {trajectory['black_hole']}")
        print(f"  Episode Length: {trajectory['episode_length']} steps")
        print(f"  Total Reward: {trajectory['total_reward']:.2f}")
        
        print(f"\nAction Statistics:")
        direction_counts = {
            'Up': trajectory['directions'].count(0),
            'Right': trajectory['directions'].count(1),
            'Down': trajectory['directions'].count(2),
            'Left': trajectory['directions'].count(3)
        }
        
        total_actions = sum(direction_counts.values())
        for direction, count in direction_counts.items():
            percentage = count / total_actions * 100
            print(f"  {direction:8s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\nThrust Statistics:")
        print(f"  Mean thrust: {np.mean(trajectory['thrusts']):.3f}")
        print(f"  Std thrust:  {np.std(trajectory['thrusts']):.3f}")
        print(f"  Min thrust:  {np.min(trajectory['thrusts']):.3f}")
        print(f"  Max thrust:  {np.max(trajectory['thrusts']):.3f}")
        
        print(f"\nTrajectory Statistics:")
        positions = np.array(trajectory['positions'])
        velocities = np.array(trajectory['velocities'])
        
        print(f"  Start position: ({positions[0, 0]:.1f}, {positions[0, 1]:.1f})")
        print(f"  End position:   ({positions[-1, 0]:.1f}, {positions[-1, 1]:.1f})")
        print(f"  Mean velocity:  {np.mean(np.linalg.norm(velocities, axis=1)):.3f}")
        print(f"  Max velocity:   {np.max(np.linalg.norm(velocities, axis=1)):.3f}")
        
        # Distance to target
        target_pos = np.array([self.env.target_planet.x, self.env.target_planet.y])
        final_distance = np.linalg.norm(positions[-1] - target_pos)
        print(f"  Final distance to target: {final_distance:.2f}")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PPO Policy Actions")
    parser.add_argument("--model-path", type=str, 
                       default="checkpoints/sb3_ppo_quick/best_model.zip",
                       help="Path to trained model")
    parser.add_argument("--output-dir", type=str,
                       default="checkpoints/sb3_ppo_quick/policy_analysis",
                       help="Directory to save visualizations")
    parser.add_argument("--n-episodes", type=int, default=5,
                       help="Number of episodes to visualize")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze action distribution over 100 episodes")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PPO POLICY VISUALIZATION")
    print("="*80)
    
    # Load policy
    visualizer = PolicyVisualizer(args.model_path)
    
    # Visualize individual episodes
    print(f"\nGenerating trajectory visualizations for {args.n_episodes} episodes...")
    for ep in range(args.n_episodes):
        print(f"\nEpisode {ep + 1}/{args.n_episodes}:")
        trajectory = visualizer.rollout_episode(deterministic=True)
        
        # Print analysis
        visualizer.print_action_analysis(trajectory)
        
        # Plot trajectory
        save_path = output_dir / f"trajectory_episode_{ep+1}.png"
        visualizer.plot_trajectory(trajectory, save_path=str(save_path))
    
    # Analyze action distribution if requested
    if args.analyze:
        stats = visualizer.analyze_action_distribution(n_episodes=100)
        
        print("\n" + "="*80)
        print("ACTION DISTRIBUTION ANALYSIS (100 episodes)")
        print("="*80)
        print(f"\nDirection Distribution:")
        for direction, count in stats['direction_counts'].items():
            print(f"  {direction}: {count}")
        
        print(f"\nThrust Statistics:")
        print(f"  Mean: {stats['mean_thrust']:.3f} ± {stats['std_thrust']:.3f}")
        
        print(f"\nPerformance:")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Black Hole Rate: {stats['black_hole_rate']:.2%}")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Mean Episode Length: {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
        
        # Plot statistics
        save_path = output_dir / "action_statistics.png"
        visualizer.plot_action_statistics(stats, save_path=str(save_path))
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*80 + "\n")

