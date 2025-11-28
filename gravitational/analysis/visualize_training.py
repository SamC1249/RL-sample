"""
Comprehensive Training Visualization Tool
Analyzes and visualizes DQN training results from checkpoints.
Similar to TensorBoard functionality but customized for this environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10


class TrainingAnalyzer:
    """Analyze and visualize training progress from checkpoints"""
    
    def __init__(self, checkpoint_dir: str, results_dir: str = None):
        """
        Args:
            checkpoint_dir: Directory containing training checkpoints
            results_dir: Directory to save analysis results
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir) if results_dir else self.checkpoint_dir.parent / "analysis" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training statistics
        self.stats = self._load_training_stats()
        self.checkpoints = self._load_checkpoints()
        
    def _load_training_stats(self) -> Dict:
        """Load training statistics from JSON file"""
        stats_path = self.checkpoint_dir / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: training_stats.json not found in {self.checkpoint_dir}")
            return {}
    
    def _load_checkpoints(self) -> List[Tuple[int, Path]]:
        """Load all checkpoint files and extract episode numbers"""
        checkpoints = []
        for ckpt_file in sorted(self.checkpoint_dir.glob("checkpoint_ep*.pt")):
            # Extract episode number from filename
            episode_num = int(ckpt_file.stem.split('_ep')[-1])
            checkpoints.append((episode_num, ckpt_file))
        
        # Sort by episode number
        checkpoints.sort(key=lambda x: x[0])
        print(f"Found {len(checkpoints)} checkpoints")
        return checkpoints
    
    def analyze_checkpoint_progression(self) -> Dict:
        """Analyze how Q-network weights change over training"""
        if len(self.checkpoints) == 0:
            print("No checkpoints found!")
            return {}
        
        weight_norms = []
        gradient_norms = []
        epsilon_values = []
        episodes = []
        
        for episode, ckpt_path in self.checkpoints:
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                
                # Calculate weight norms
                q_network_state = checkpoint['q_network_state_dict']
                total_norm = 0.0
                for param_name, param in q_network_state.items():
                    if 'weight' in param_name:
                        total_norm += torch.norm(param).item() ** 2
                weight_norms.append(np.sqrt(total_norm))
                
                # Get epsilon
                epsilon_values.append(checkpoint.get('epsilon', 0.0))
                episodes.append(episode)
                
            except Exception as e:
                print(f"Error loading checkpoint {ckpt_path}: {e}")
                continue
        
        return {
            'episodes': episodes,
            'weight_norms': weight_norms,
            'epsilon_values': epsilon_values
        }
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive training dashboard"""
        print("\n" + "="*80)
        print("Creating Comprehensive Training Dashboard")
        print("="*80)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Load checkpoint progression data
        ckpt_data = self.analyze_checkpoint_progression()
        
        # 1. Training Statistics Summary (Text Box)
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis('off')
        
        summary_text = self._create_summary_text()
        ax_summary.text(0.05, 0.5, summary_text, transform=ax_summary.transAxes,
                       fontsize=11, verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_summary.set_title('Training Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 2. Weight Norms Over Time
        if ckpt_data.get('weight_norms'):
            ax1 = fig.add_subplot(gs[1, 0])
            ax1.plot(ckpt_data['episodes'], ckpt_data['weight_norms'], 
                    marker='o', linewidth=2, markersize=4, color='blue')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('L2 Norm of Weights')
            ax1.set_title('Q-Network Weight Magnitude')
            ax1.grid(alpha=0.3)
        
        # 3. Epsilon Decay
        if ckpt_data.get('epsilon_values'):
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.plot(ckpt_data['episodes'], ckpt_data['epsilon_values'], 
                    marker='o', linewidth=2, markersize=4, color='purple')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            ax2.set_title('Exploration Rate Decay')
            ax2.set_ylim([0, 1.1])
            ax2.grid(alpha=0.3)
        
        # 4. Checkpoint Distribution
        ax3 = fig.add_subplot(gs[1, 2])
        if ckpt_data.get('episodes'):
            ax3.hist(ckpt_data['episodes'], bins=20, color='green', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Episode Number')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Checkpoint Distribution')
            ax3.grid(alpha=0.3, axis='y')
        
        # 5. Problem Analysis - Reward Distribution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_reward_analysis(ax4)
        
        # 6. Problem Analysis - Success vs Black Hole
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_outcome_analysis(ax5)
        
        # 7. Learning Progress Indicator
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_learning_progress(ax6)
        
        # 8. Recommendations Box
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        recommendations = self._generate_recommendations()
        ax7.text(0.05, 0.5, recommendations, transform=ax7.transAxes,
                fontsize=10, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
        ax7.set_title('Recommendations for Improvement', fontsize=14, 
                     fontweight='bold', pad=20, color='red')
        
        # Main title
        fig.suptitle('DQN Training Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = self.results_dir / "training_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Dashboard saved to: {output_path}")
        plt.close()
        
        # Create additional detailed plots
        self._create_detailed_plots()
    
    def _create_summary_text(self) -> str:
        """Create summary statistics text"""
        if not self.stats:
            return "No training statistics available"
        
        text = (
            f"Total Episodes: {self.stats.get('total_episodes', 'N/A')}\n"
            f"Average Reward (Overall): {self.stats.get('avg_reward', 0):.2f}\n"
            f"Average Reward (Recent 100): {self.stats.get('avg_reward_recent', 0):.2f}\n"
            f"Average Episode Length: {self.stats.get('avg_length', 0):.1f} steps\n"
            f"Success Rate (Overall): {self.stats.get('success_rate', 0)*100:.2f}%\n"
            f"Success Rate (Recent): {self.stats.get('success_rate_recent', 0)*100:.2f}%\n"
            f"Black Hole Death Rate (Overall): {self.stats.get('black_hole_rate', 0)*100:.2f}%\n"
            f"Black Hole Death Rate (Recent): {self.stats.get('black_hole_rate_recent', 0)*100:.2f}%\n"
            f"Final Epsilon: {self.stats.get('current_epsilon', 0):.4f}"
        )
        return text
    
    def _plot_reward_analysis(self, ax):
        """Analyze reward distribution"""
        if not self.stats:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return
        
        # Create reward breakdown
        avg_reward = self.stats.get('avg_reward', 0)
        recent_reward = self.stats.get('avg_reward_recent', 0)
        
        categories = ['Overall\nAverage', 'Recent 100\nAverage']
        values = [avg_reward, recent_reward]
        colors = ['red' if v < -1000 else 'orange' if v < 0 else 'green' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel('Average Reward')
        ax.set_title('Reward Analysis')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.0f}', ha='center', va='bottom' if height > 0 else 'top')
    
    def _plot_outcome_analysis(self, ax):
        """Analyze episode outcomes"""
        if not self.stats:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return
        
        success_rate = self.stats.get('success_rate_recent', 0) * 100
        black_hole_rate = self.stats.get('black_hole_rate_recent', 0) * 100
        timeout_rate = 100 - success_rate - black_hole_rate
        
        outcomes = ['Success', 'Black Hole\nDeath', 'Timeout']
        percentages = [success_rate, black_hole_rate, timeout_rate]
        colors = ['green', 'red', 'orange']
        
        wedges, texts, autotexts = ax.pie(percentages, labels=outcomes, colors=colors,
                                          autopct='%1.1f%%', startangle=90)
        ax.set_title('Episode Outcomes (Recent 100)')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_learning_progress(self, ax):
        """Plot learning progress indicator"""
        if not self.stats:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            return
        
        # Calculate learning score (0-100)
        success_rate = self.stats.get('success_rate_recent', 0)
        black_hole_rate = self.stats.get('black_hole_rate_recent', 0)
        
        # Score: high success = good, high black hole = bad
        learning_score = max(0, min(100, success_rate * 100 - black_hole_rate * 50))
        
        # Create gauge-like visualization
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # Background arc
        ax.plot(theta, r, 'k-', linewidth=20, alpha=0.2)
        
        # Progress arc
        progress_theta = np.linspace(0, np.pi * (learning_score / 100), 100)
        color = 'red' if learning_score < 30 else 'orange' if learning_score < 60 else 'green'
        ax.plot(progress_theta, r[:len(progress_theta)], color=color, linewidth=20)
        
        # Score text
        ax.text(np.pi/2, 0.5, f'{learning_score:.1f}', ha='center', va='center',
               fontsize=24, fontweight='bold')
        ax.text(np.pi/2, 0.2, 'Learning Score', ha='center', va='center', fontsize=12)
        
        ax.set_xlim([0, np.pi])
        ax.set_ylim([0, 1.2])
        ax.axis('off')
        ax.set_title('Overall Learning Progress')
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on training analysis"""
        if not self.stats:
            return "No data available for recommendations"
        
        recommendations = []
        
        # Check success rate
        success_rate = self.stats.get('success_rate_recent', 0)
        if success_rate < 0.01:
            recommendations.append(
                "⚠ CRITICAL: 0% success rate - Agent never reaches target!\n"
                "  → Check reward shaping: Is reaching target rewarding enough?\n"
                "  → Verify terminal condition detection in training loop (line 240-243 in train.py)"
            )
        
        # Check black hole rate
        black_hole_rate = self.stats.get('black_hole_rate_recent', 0)
        if black_hole_rate > 0.5:
            recommendations.append(
                f"⚠ HIGH: {black_hole_rate*100:.1f}% black hole death rate\n"
                "  → Increase penalty for approaching black hole\n"
                "  → Add distance-to-black-hole to state space\n"
                "  → Consider reward shaping: penalize getting closer to black hole"
            )
        
        # Check reward magnitude
        avg_reward = self.stats.get('avg_reward_recent', 0)
        if avg_reward < -1000:
            recommendations.append(
                f"⚠ ISSUE: Very negative rewards ({avg_reward:.0f})\n"
                "  → Gravity cost dominates (k=0.1 too small compared to gravity penalties)\n"
                "  → Consider: Increase k, normalize rewards, or add positive shaping\n"
                "  → Bug: Reward calculation may not properly detect target arrival"
            )
        
        # Check epsilon
        epsilon = self.stats.get('current_epsilon', 0)
        if epsilon > 0.1 and self.stats.get('total_episodes', 0) > 1000:
            recommendations.append(
                f"⚠ INFO: Epsilon still at {epsilon:.3f} after many episodes\n"
                "  → Consider faster epsilon decay for more exploitation"
            )
        
        if not recommendations:
            recommendations.append("✓ Training metrics look reasonable - continue monitoring")
        
        return "\n\n".join(recommendations)
    
    def _create_detailed_plots(self):
        """Create additional detailed analysis plots"""
        print("\nCreating detailed analysis plots...")
        
        # 1. Checkpoint weight evolution
        self._plot_weight_evolution()
        
        # 2. Problem diagnosis
        self._plot_problem_diagnosis()
        
        print("✓ All analysis plots created successfully!")
    
    def _plot_weight_evolution(self):
        """Plot detailed weight evolution across checkpoints"""
        if len(self.checkpoints) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        episodes = []
        layer_norms = {f'layer_{i}': [] for i in range(3)}  # Assuming 3 layers
        
        for episode, ckpt_path in self.checkpoints:
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                q_network_state = checkpoint['q_network_state_dict']
                
                episodes.append(episode)
                
                # Calculate norms for each layer
                layer_idx = 0
                for param_name, param in q_network_state.items():
                    if 'weight' in param_name:
                        norm = torch.norm(param).item()
                        layer_norms[f'layer_{layer_idx}'].append(norm)
                        layer_idx += 1
                        if layer_idx >= 3:
                            break
                
            except Exception as e:
                continue
        
        # Plot each layer's weight evolution
        for idx, (layer_name, norms) in enumerate(layer_norms.items()):
            if norms:
                ax = axes[idx // 2, idx % 2]
                ax.plot(episodes, norms, marker='o', linewidth=2, markersize=4)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Weight Norm')
                ax.set_title(f'Weight Evolution - {layer_name.replace("_", " ").title()}')
                ax.grid(alpha=0.3)
        
        # Use last subplot for combined view
        ax = axes[1, 1]
        for layer_name, norms in layer_norms.items():
            if norms:
                ax.plot(episodes, norms, marker='o', linewidth=2, 
                       markersize=3, label=layer_name, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Weight Norm')
        ax.set_title('All Layers Combined')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.suptitle('Q-Network Weight Evolution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.results_dir / "weight_evolution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  → Weight evolution plot: {output_path}")
        plt.close()
    
    def _plot_problem_diagnosis(self):
        """Create diagnostic plots for identified problems"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Problem 1: Reward Structure
        ax1 = axes[0, 0]
        ax1.text(0.5, 0.9, 'Problem: Reward Structure', ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax1.transAxes)
        
        problem_text = (
            "Current reward formula:\n"
            "r = -(||g|| · (1 - cos θ)) + k*T\n\n"
            "Issues:\n"
            "• Gravity cost dominates (||g|| can be large)\n"
            "• k=0.1 is too small (thrust barely rewarded)\n"
            "• No explicit reward for approaching target\n"
            "• Black hole penalty (-100) only at death\n\n"
            "Suggested fix:\n"
            "• Add distance-to-target shaping\n"
            "• Increase k or normalize gravity cost\n"
            "• Add progressive black hole warnings"
        )
        ax1.text(0.1, 0.7, problem_text, ha='left', va='top',
                fontsize=10, family='monospace', transform=ax1.transAxes)
        ax1.axis('off')
        
        # Problem 2: Terminal Detection Bug
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.9, 'Bug: Terminal Condition Detection', ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax2.transAxes, color='red')
        
        bug_text = (
            "train.py lines 240-243:\n"
            "if reward == 0:  # Target planet reward\n"
            "    success = True\n\n"
            "Problem:\n"
            "• 'reward' is the STEP reward (gravity + thrust)\n"
            "• Step reward is NEVER exactly 0\n"
            "• Target planet gives 0 BONUS, not 0 total\n"
            "• Success never detected!\n\n"
            "Fix:\n"
            "Check info dict or terminated flag reason:\n"
            "if terminated and info['distance_to_target'] < 2.0:\n"
            "    success = True"
        )
        ax2.text(0.1, 0.7, bug_text, ha='left', va='top',
                fontsize=9, family='monospace', transform=ax2.transAxes)
        ax2.axis('off')
        
        # Problem 3: State Space
        ax3 = axes[1, 0]
        ax3.text(0.5, 0.9, 'Issue: Limited State Information', ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax3.transAxes)
        
        state_text = (
            "Current state: [x, y, vx, vy]\n\n"
            "Missing information:\n"
            "• Distance to target\n"
            "• Distance to black hole\n"
            "• Direction to target\n"
            "• Gravity magnitude at position\n\n"
            "Agent can't easily learn to:\n"
            "• Navigate toward target\n"
            "• Avoid black hole danger zone\n\n"
            "Suggested additions:\n"
            "state = [x, y, vx, vy, \n"
            "         dist_to_target, angle_to_target,\n"
            "         dist_to_black_hole, ||g||]"
        )
        ax3.text(0.1, 0.7, state_text, ha='left', va='top',
                fontsize=10, family='monospace', transform=ax3.transAxes)
        ax3.axis('off')
        
        # Problem 4: Statistics Summary
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.9, 'Training Statistics Analysis', ha='center', va='top',
                fontsize=14, fontweight='bold', transform=ax4.transAxes)
        
        if self.stats:
            stats_text = (
                f"Episodes: {self.stats.get('total_episodes', 'N/A')}\n"
                f"Avg Reward: {self.stats.get('avg_reward', 0):.1f}\n"
                f"Recent Reward: {self.stats.get('avg_reward_recent', 0):.1f}\n\n"
                f"Success Rate: {self.stats.get('success_rate_recent', 0)*100:.1f}%\n"
                f"Black Hole Rate: {self.stats.get('black_hole_rate_recent', 0)*100:.1f}%\n"
                f"Timeout Rate: {(1-self.stats.get('success_rate_recent', 0)-self.stats.get('black_hole_rate_recent', 0))*100:.1f}%\n\n"
                "Interpretation:\n"
                "• 0% success = never reaches target\n"
                "• ~70% black hole = poor navigation\n"
                "• Negative rewards = punishment-heavy\n\n"
                "Root cause: Combination of\n"
                "1. Reward structure issues\n"
                "2. Terminal detection bug\n"
                "3. Limited state information"
            )
        else:
            stats_text = "No statistics available"
        
        ax4.text(0.1, 0.7, stats_text, ha='left', va='top',
                fontsize=10, family='monospace', transform=ax4.transAxes)
        ax4.axis('off')
        
        plt.suptitle('Problem Diagnosis & Recommendations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.results_dir / "problem_diagnosis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  → Problem diagnosis: {output_path}")
        plt.close()
    
    def generate_report(self):
        """Generate a text report with findings"""
        report_path = self.results_dir / "analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DQN TRAINING ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("TRAINING STATISTICS\n")
            f.write("-"*80 + "\n")
            if self.stats:
                for key, value in self.stats.items():
                    f.write(f"{key:30s}: {value}\n")
            else:
                f.write("No statistics available\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("IDENTIFIED PROBLEMS\n")
            f.write("="*80 + "\n\n")
            
            f.write(self._generate_recommendations())
            
            f.write("\n\n" + "="*80 + "\n")
            f.write("CHECKPOINTS ANALYZED\n")
            f.write("="*80 + "\n")
            f.write(f"Total checkpoints: {len(self.checkpoints)}\n")
            if self.checkpoints:
                f.write(f"First checkpoint: Episode {self.checkpoints[0][0]}\n")
                f.write(f"Last checkpoint: Episode {self.checkpoints[-1][0]}\n")
        
        print(f"\n✓ Analysis report saved to: {report_path}")


def main():
    """Main analysis function"""
    import sys
    
    # Default checkpoint directory
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "dqn"
    
    # Allow command line override
    if len(sys.argv) > 1:
        checkpoint_dir = Path(sys.argv[1])
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        print("Usage: python visualize_training.py [checkpoint_dir]")
        return
    
    print("\n" + "="*80)
    print("DQN TRAINING ANALYSIS TOOL")
    print("="*80)
    print(f"\nAnalyzing checkpoints from: {checkpoint_dir}")
    
    # Create analyzer
    analyzer = TrainingAnalyzer(str(checkpoint_dir))
    
    # Generate visualizations
    analyzer.create_comprehensive_dashboard()
    
    # Generate text report
    analyzer.generate_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {analyzer.results_dir}")
    print("\nGenerated files:")
    print("  • training_dashboard.png    - Main dashboard with all metrics")
    print("  • weight_evolution.png      - Q-network weight changes")
    print("  • problem_diagnosis.png     - Detailed problem analysis")
    print("  • analysis_report.txt       - Text summary report")
    print("\n")


if __name__ == "__main__":
    main()

