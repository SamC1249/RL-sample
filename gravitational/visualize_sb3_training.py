"""
Visualize Stable-Baselines3 PPO Training Results
Reads TensorBoard logs and creates custom plots for PPO training.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import os
from typing import Dict, List, Tuple
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10


class SB3TrainingVisualizer:
    """Visualize training from Stable-Baselines3 TensorBoard logs"""
    
    def __init__(self, checkpoint_dir: str, output_dir: str = None):
        """
        Args:
            checkpoint_dir: Directory containing SB3 checkpoints and logs
            output_dir: Directory to save visualizations
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir) if output_dir else self.checkpoint_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find TensorBoard logs
        self.log_dir = self.checkpoint_dir / "logs"
        self.events = self._load_tensorboard_events()
        
        # Load evaluation data if available
        self.eval_data = self._load_evaluation_data()
        
    def _load_tensorboard_events(self) -> Dict:
        """Load all scalar data from TensorBoard event files"""
        print(f"Loading TensorBoard events from {self.log_dir}...")
        
        data = {
            'rollout': {},
            'train': {},
            'eval': {},
            'time': {}
        }
        
        # Find all event files
        event_files = list(self.log_dir.rglob("events.out.tfevents.*"))
        
        if not event_files:
            print(f"Warning: No TensorBoard event files found in {self.log_dir}")
            return data
        
        print(f"Found {len(event_files)} event file(s)")
        
        for event_file in event_files:
            print(f"  Loading: {event_file.name}")
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            
            # Get all scalar tags
            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                # Categorize by prefix
                if tag.startswith('rollout/'):
                    key = tag.replace('rollout/', '')
                    data['rollout'][key] = {'steps': steps, 'values': values}
                elif tag.startswith('train/'):
                    key = tag.replace('train/', '')
                    data['train'][key] = {'steps': steps, 'values': values}
                elif tag.startswith('eval/'):
                    key = tag.replace('eval/', '')
                    data['eval'][key] = {'steps': steps, 'values': values}
                elif tag.startswith('time/'):
                    key = tag.replace('time/', '')
                    data['time'][key] = {'steps': steps, 'values': values}
        
        print("TensorBoard data loaded successfully!")
        return data
    
    def _load_evaluation_data(self) -> Dict:
        """Load evaluation results if available"""
        eval_file = self.log_dir / "evaluations.npz"
        if eval_file.exists():
            data = np.load(eval_file)
            return {
                'timesteps': data['timesteps'],
                'results': data['results'],
                'ep_lengths': data['ep_lengths']
            }
        return {}
    
    def plot_training_overview(self, save_path: str = None):
        """Create comprehensive training overview plot"""
        if save_path is None:
            save_path = self.output_dir / "training_overview.png"
        
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Episode Reward
        ax1 = fig.add_subplot(gs[0, 0])
        if 'ep_rew_mean' in self.events['rollout']:
            data = self.events['rollout']['ep_rew_mean']
            ax1.plot(data['steps'], data['values'], linewidth=2, color='blue', alpha=0.7)
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Mean Episode Reward')
            ax1.set_title('Episode Reward Over Time')
            ax1.grid(alpha=0.3)
        
        # 2. Success Rate
        ax2 = fig.add_subplot(gs[0, 1])
        if 'success_rate' in self.events['rollout']:
            data = self.events['rollout']['success_rate']
            ax2.plot(data['steps'], data['values'], linewidth=2, color='green', alpha=0.7)
            ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Target')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Success Rate')
            ax2.set_title('Success Rate Over Time')
            ax2.set_ylim([0, 1])
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # 3. Black Hole Rate
        ax3 = fig.add_subplot(gs[0, 2])
        if 'black_hole_rate' in self.events['rollout']:
            data = self.events['rollout']['black_hole_rate']
            ax3.plot(data['steps'], data['values'], linewidth=2, color='red', alpha=0.7)
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Black Hole Death Rate')
            ax3.set_title('Black Hole Deaths Over Time')
            ax3.set_ylim([0, 1])
            ax3.grid(alpha=0.3)
        
        # 4. Episode Length
        ax4 = fig.add_subplot(gs[1, 0])
        if 'ep_len_mean' in self.events['rollout']:
            data = self.events['rollout']['ep_len_mean']
            ax4.plot(data['steps'], data['values'], linewidth=2, color='purple', alpha=0.7)
            ax4.set_xlabel('Timesteps')
            ax4.set_ylabel('Mean Episode Length')
            ax4.set_title('Episode Length Over Time')
            ax4.grid(alpha=0.3)
        
        # 5. Policy Loss
        ax5 = fig.add_subplot(gs[1, 1])
        if 'policy_gradient_loss' in self.events['train']:
            data = self.events['train']['policy_gradient_loss']
            ax5.plot(data['steps'], data['values'], linewidth=2, color='orange', alpha=0.7)
            ax5.set_xlabel('Timesteps')
            ax5.set_ylabel('Policy Gradient Loss')
            ax5.set_title('Policy Loss Over Time')
            ax5.grid(alpha=0.3)
        
        # 6. Value Loss
        ax6 = fig.add_subplot(gs[1, 2])
        if 'value_loss' in self.events['train']:
            data = self.events['train']['value_loss']
            ax6.plot(data['steps'], data['values'], linewidth=2, color='brown', alpha=0.7)
            ax6.set_xlabel('Timesteps')
            ax6.set_ylabel('Value Loss')
            ax6.set_title('Value Function Loss Over Time')
            ax6.grid(alpha=0.3)
        
        # 7. Entropy Loss
        ax7 = fig.add_subplot(gs[2, 0])
        if 'entropy_loss' in self.events['train']:
            data = self.events['train']['entropy_loss']
            ax7.plot(data['steps'], data['values'], linewidth=2, color='teal', alpha=0.7)
            ax7.set_xlabel('Timesteps')
            ax7.set_ylabel('Entropy Loss')
            ax7.set_title('Policy Entropy Over Time')
            ax7.grid(alpha=0.3)
        
        # 8. KL Divergence
        ax8 = fig.add_subplot(gs[2, 1])
        if 'approx_kl' in self.events['train']:
            data = self.events['train']['approx_kl']
            ax8.plot(data['steps'], data['values'], linewidth=2, color='magenta', alpha=0.7)
            ax8.set_xlabel('Timesteps')
            ax8.set_ylabel('Approximate KL Divergence')
            ax8.set_title('KL Divergence Over Time')
            ax8.grid(alpha=0.3)
        
        # 9. Clip Fraction
        ax9 = fig.add_subplot(gs[2, 2])
        if 'clip_fraction' in self.events['train']:
            data = self.events['train']['clip_fraction']
            ax9.plot(data['steps'], data['values'], linewidth=2, color='cyan', alpha=0.7)
            ax9.set_xlabel('Timesteps')
            ax9.set_ylabel('Clip Fraction')
            ax9.set_title('PPO Clipping Fraction Over Time')
            ax9.grid(alpha=0.3)
        
        # 10. Learning Rate
        ax10 = fig.add_subplot(gs[3, 0])
        if 'learning_rate' in self.events['train']:
            data = self.events['train']['learning_rate']
            ax10.plot(data['steps'], data['values'], linewidth=2, color='navy', alpha=0.7)
            ax10.set_xlabel('Timesteps')
            ax10.set_ylabel('Learning Rate')
            ax10.set_title('Learning Rate Schedule')
            ax10.grid(alpha=0.3)
        
        # 11. Explained Variance
        ax11 = fig.add_subplot(gs[3, 1])
        if 'explained_variance' in self.events['train']:
            data = self.events['train']['explained_variance']
            ax11.plot(data['steps'], data['values'], linewidth=2, color='olive', alpha=0.7)
            ax11.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax11.set_xlabel('Timesteps')
            ax11.set_ylabel('Explained Variance')
            ax11.set_title('Value Function Explained Variance')
            ax11.grid(alpha=0.3)
        
        # 12. FPS
        ax12 = fig.add_subplot(gs[3, 2])
        if 'fps' in self.events['time']:
            data = self.events['time']['fps']
            ax12.plot(data['steps'], data['values'], linewidth=2, color='darkgreen', alpha=0.7)
            ax12.set_xlabel('Timesteps')
            ax12.set_ylabel('Frames Per Second')
            ax12.set_title('Training Speed (FPS)')
            ax12.grid(alpha=0.3)
        
        plt.suptitle('PPO Training Overview', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training overview to: {save_path}")
        plt.close()
    
    def plot_performance_summary(self, save_path: str = None):
        """Create focused plot on agent performance metrics"""
        if save_path is None:
            save_path = self.output_dir / "performance_summary.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Reward with smoothing
        ax1 = axes[0, 0]
        if 'ep_rew_mean' in self.events['rollout']:
            data = self.events['rollout']['ep_rew_mean']
            steps = np.array(data['steps'])
            values = np.array(data['values'])
            
            # Raw data
            ax1.plot(steps, values, alpha=0.3, color='blue', label='Raw')
            
            # Smoothed (moving average)
            if len(values) > 10:
                window = min(50, len(values) // 10)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax1.plot(smooth_steps, smoothed, linewidth=3, color='darkblue', label=f'Smoothed (window={window})')
            
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Mean Episode Reward')
            ax1.set_title('Episode Reward Progression')
            ax1.legend()
            ax1.grid(alpha=0.3)
        
        # 2. Success vs Black Hole Rate
        ax2 = axes[0, 1]
        if 'success_rate' in self.events['rollout'] and 'black_hole_rate' in self.events['rollout']:
            success_data = self.events['rollout']['success_rate']
            black_hole_data = self.events['rollout']['black_hole_rate']
            
            ax2.plot(success_data['steps'], success_data['values'], 
                    linewidth=2, color='green', label='Success Rate', alpha=0.7)
            ax2.plot(black_hole_data['steps'], black_hole_data['values'], 
                    linewidth=2, color='red', label='Black Hole Rate', alpha=0.7)
            
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Rate')
            ax2.set_title('Success vs Black Hole Death Rate')
            ax2.set_ylim([0, 1])
            ax2.legend()
            ax2.grid(alpha=0.3)
        
        # 3. Episode reward distribution over time (heatmap or box plot)
        ax3 = axes[1, 0]
        if 'ep_reward' in self.events['rollout']:
            data = self.events['rollout']['ep_reward']
            steps = np.array(data['steps'])
            values = np.array(data['values'])
            
            # Create bins for different training phases
            n_bins = 10
            step_bins = np.linspace(steps.min(), steps.max(), n_bins + 1)
            bin_rewards = [[] for _ in range(n_bins)]
            
            for step, value in zip(steps, values):
                bin_idx = min(np.searchsorted(step_bins, step) - 1, n_bins - 1)
                if bin_idx >= 0:
                    bin_rewards[bin_idx].append(value)
            
            # Box plot
            positions = [(step_bins[i] + step_bins[i+1])/2 for i in range(n_bins)]
            ax3.boxplot([r for r in bin_rewards if r], positions=positions, widths=step_bins[1]-step_bins[0]*0.8)
            ax3.set_xlabel('Timesteps')
            ax3.set_ylabel('Episode Reward')
            ax3.set_title('Episode Reward Distribution Over Training')
            ax3.grid(alpha=0.3)
        
        # 4. Training efficiency (reward per timestep)
        ax4 = axes[1, 1]
        if 'ep_rew_mean' in self.events['rollout'] and 'ep_len_mean' in self.events['rollout']:
            reward_data = self.events['rollout']['ep_rew_mean']
            length_data = self.events['rollout']['ep_len_mean']
            
            # Calculate reward per step
            if len(reward_data['steps']) == len(length_data['steps']):
                efficiency = np.array(reward_data['values']) / np.array(length_data['values'])
                ax4.plot(reward_data['steps'], efficiency, linewidth=2, color='purple', alpha=0.7)
                ax4.set_xlabel('Timesteps')
                ax4.set_ylabel('Reward per Step')
                ax4.set_title('Training Efficiency (Reward/Step)')
                ax4.grid(alpha=0.3)
        
        plt.suptitle('PPO Performance Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance summary to: {save_path}")
        plt.close()
    
    def generate_report(self):
        """Generate text report with key statistics"""
        report_path = self.output_dir / "training_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PPO TRAINING REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Final statistics
            f.write("FINAL STATISTICS:\n")
            f.write("-"*80 + "\n")
            
            if 'ep_rew_mean' in self.events['rollout']:
                final_reward = self.events['rollout']['ep_rew_mean']['values'][-1]
                f.write(f"Final Mean Episode Reward: {final_reward:.2f}\n")
            
            if 'success_rate' in self.events['rollout']:
                final_success = self.events['rollout']['success_rate']['values'][-1]
                f.write(f"Final Success Rate: {final_success:.2%}\n")
            
            if 'black_hole_rate' in self.events['rollout']:
                final_bh = self.events['rollout']['black_hole_rate']['values'][-1]
                f.write(f"Final Black Hole Rate: {final_bh:.2%}\n")
            
            if 'ep_len_mean' in self.events['rollout']:
                final_len = self.events['rollout']['ep_len_mean']['values'][-1]
                f.write(f"Final Mean Episode Length: {final_len:.0f}\n")
            
            f.write("\n")
            
            # Training progress
            f.write("TRAINING PROGRESS:\n")
            f.write("-"*80 + "\n")
            
            if 'success_rate' in self.events['rollout']:
                success_vals = self.events['rollout']['success_rate']['values']
                initial_success = success_vals[0] if success_vals else 0
                final_success = success_vals[-1] if success_vals else 0
                improvement = final_success - initial_success
                f.write(f"Success Rate: {initial_success:.2%} → {final_success:.2%} (Δ {improvement:+.2%})\n")
            
            if 'ep_rew_mean' in self.events['rollout']:
                reward_vals = self.events['rollout']['ep_rew_mean']['values']
                initial_reward = reward_vals[0] if reward_vals else 0
                final_reward = reward_vals[-1] if reward_vals else 0
                improvement = final_reward - initial_reward
                f.write(f"Mean Reward: {initial_reward:.2f} → {final_reward:.2f} (Δ {improvement:+.2f})\n")
            
            f.write("\n")
            
            # Best performance
            f.write("BEST PERFORMANCE:\n")
            f.write("-"*80 + "\n")
            
            if 'ep_rew_mean' in self.events['rollout']:
                rewards = self.events['rollout']['ep_rew_mean']['values']
                steps = self.events['rollout']['ep_rew_mean']['steps']
                best_idx = np.argmax(rewards)
                f.write(f"Best Reward: {rewards[best_idx]:.2f} at step {steps[best_idx]}\n")
            
            if 'success_rate' in self.events['rollout']:
                success = self.events['rollout']['success_rate']['values']
                steps = self.events['rollout']['success_rate']['steps']
                best_idx = np.argmax(success)
                f.write(f"Best Success Rate: {success[best_idx]:.2%} at step {steps[best_idx]}\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
        
        print(f"Saved training report to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize SB3 PPO Training")
    parser.add_argument("--checkpoint-dir", type=str, 
                       default="checkpoints/sb3_ppo_quick",
                       help="Directory containing SB3 checkpoints and logs")
    parser.add_argument("--output-dir", type=str,
                       help="Directory to save visualizations (default: checkpoint_dir/visualizations)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("SB3 PPO Training Visualization")
    print("="*80)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    
    # Create visualizer
    visualizer = SB3TrainingVisualizer(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer.plot_training_overview()
    visualizer.plot_performance_summary()
    visualizer.generate_report()
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"Results saved to: {visualizer.output_dir}")
    print("="*80)

