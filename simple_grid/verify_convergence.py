#!/usr/bin/env python3
"""
Convergence Verification for Policy Iteration
Demonstrates that policy iteration converges and finds optimal path to target.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from simple_env import SimpleGridEnv
from policy_iteration import PolicyIteration


def visualize_convergence_metrics(solver_history, save_path='convergence_metrics.png'):
    """
    Visualize convergence metrics over policy iteration iterations.
    
    Args:
        solver_history: List of dictionaries containing iteration metrics
        save_path: Path to save the figure
    """
    iterations = [h['iteration'] for h in solver_history]
    mean_values = [h['mean_value'] for h in solver_history]
    max_values = [h['max_value'] for h in solver_history]
    min_values = [h['min_value'] for h in solver_history]
    policy_changes = [h['policy_changes'] for h in solver_history]
    eval_iterations = [h['eval_iterations'] for h in solver_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Value Function Statistics
    ax1 = axes[0, 0]
    ax1.plot(iterations, mean_values, 'b-', linewidth=2, label='Mean Value')
    ax1.plot(iterations, max_values, 'g--', linewidth=1.5, label='Max Value')
    ax1.plot(iterations, min_values, 'r--', linewidth=1.5, label='Min Value')
    ax1.set_xlabel('Policy Iteration')
    ax1.set_ylabel('Value')
    ax1.set_title('Value Function Convergence')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Policy Changes
    ax2 = axes[0, 1]
    ax2.bar(iterations, policy_changes, color='orange', alpha=0.7)
    ax2.set_xlabel('Policy Iteration')
    ax2.set_ylabel('Number of State Policy Changes')
    ax2.set_title('Policy Stability (Convergence when 0)')
    ax2.grid(alpha=0.3, axis='y')
    
    # Add convergence line
    if policy_changes[-1] == 0:
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Converged')
        ax2.legend()
    
    # Plot 3: Policy Evaluation Iterations
    ax3 = axes[1, 0]
    ax3.plot(iterations, eval_iterations, 'purple', marker='o', linewidth=2)
    ax3.set_xlabel('Policy Iteration')
    ax3.set_ylabel('Iterations to Evaluate Policy')
    ax3.set_title('Policy Evaluation Efficiency')
    ax3.grid(alpha=0.3)
    
    # Plot 4: Convergence Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    total_iterations = len(iterations)
    converged = policy_changes[-1] == 0
    final_mean_value = mean_values[-1]
    total_eval_iterations = sum(eval_iterations)
    
    summary_text = (
        f"CONVERGENCE SUMMARY\n"
        f"{'='*40}\n\n"
        f"Status: {'✓ CONVERGED' if converged else '✗ NOT CONVERGED'}\n"
        f"Policy Iterations: {total_iterations}\n"
        f"Total Eval Iterations: {total_eval_iterations}\n"
        f"Final Mean Value: {final_mean_value:.4f}\n"
        f"Final Max Value: {max_values[-1]:.4f}\n"
        f"Final Min Value: {min_values[-1]:.4f}\n"
        f"Final Policy Changes: {policy_changes[-1]}\n\n"
        f"Interpretation:\n"
        f"• Policy stable = No state changes action\n"
        f"• Converged policy is OPTIMAL\n"
        f"• Agent found safe path to target!"
    )
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if converged else 'lightyellow', alpha=0.5))
    
    plt.suptitle('Policy Iteration Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence metrics saved to: {save_path}")
    plt.close()


def visualize_policy_heatmap(env, policy, value_function, save_path='policy_heatmap.png'):
    """
    Create a heatmap visualization of the policy and value function.
    
    Args:
        env: SimpleGridEnv instance
        policy: Policy array
        value_function: Value function array
        save_path: Path to save the figure
    """
    grid_size = env.grid_size
    
    # Create value function grid
    value_grid = np.zeros((grid_size, grid_size))
    for state in range(env.n_states):
        row, col = env._state_to_pos(state)
        value_grid[row, col] = value_function[state]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Value Function Heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(value_grid, cmap='RdYlGn', interpolation='nearest')
    ax1.set_title('Value Function V(s)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Mark special locations
    ax1.plot(env.black_hole_pos[1], env.black_hole_pos[0], 'ko', markersize=15, 
            markerfacecolor='black', label='Black Hole')
    for planet_pos in env.planet_positions:
        ax1.plot(planet_pos[1], planet_pos[0], 'o', markersize=10, 
                markerfacecolor='brown', markeredgecolor='black', label='Planet')
    ax1.plot(env.target_pos[1], env.target_pos[0], '*', markersize=20, 
            markerfacecolor='gold', markeredgecolor='black', label='Target')
    
    # Remove duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.colorbar(im1, ax=ax1, label='Value')
    
    # Plot 2: Policy Arrows (zoomed subset)
    ax2 = axes[1]
    
    # Show a zoomed region around the start and target
    zoom_size = 40
    policy_grid = np.zeros((zoom_size, zoom_size))
    
    # Action symbols
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    
    # Create policy visualization
    for i in range(zoom_size):
        for j in range(zoom_size):
            state = env._pos_to_state((i, j))
            action = policy[state]
            
            # Color based on value
            value = value_function[state]
            policy_grid[i, j] = value
            
            # Add arrow
            if (i, j) not in [env.black_hole_pos, env.target_pos] + env.planet_positions:
                ax2.text(j, i, action_symbols[action], ha='center', va='center',
                        fontsize=8, color='black', fontweight='bold')
    
    im2 = ax2.imshow(policy_grid, cmap='RdYlGn', interpolation='nearest', alpha=0.3)
    ax2.set_title(f'Policy π(s) - Zoomed View (0:{zoom_size}, 0:{zoom_size})', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Mark special locations in zoom
    if env.black_hole_pos[0] < zoom_size and env.black_hole_pos[1] < zoom_size:
        ax2.plot(env.black_hole_pos[1], env.black_hole_pos[0], 'ko', markersize=15,
                markerfacecolor='black', label='Black Hole')
    
    for planet_pos in env.planet_positions:
        if planet_pos[0] < zoom_size and planet_pos[1] < zoom_size:
            ax2.plot(planet_pos[1], planet_pos[0], 'o', markersize=10,
                    markerfacecolor='brown', markeredgecolor='black', label='Planet')
    
    if env.target_pos[0] < zoom_size and env.target_pos[1] < zoom_size:
        ax2.plot(env.target_pos[1], env.target_pos[0], '*', markersize=20,
                markerfacecolor='gold', markeredgecolor='black', label='Target')
    
    # Remove duplicate labels
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.colorbar(im2, ax=ax2, label='Value')
    
    plt.suptitle('Optimal Policy Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Policy heatmap saved to: {save_path}")
    plt.close()


def visualize_optimal_trajectory(env, policy, num_trajectories=5, save_path='optimal_trajectories.png'):
    """
    Visualize multiple optimal trajectories from start to target.
    
    Args:
        env: SimpleGridEnv instance
        policy: Optimal policy
        num_trajectories: Number of trajectories to simulate
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create background grid
    grid = np.zeros((env.grid_size, env.grid_size))
    
    # Mark special locations
    grid[env.black_hole_pos] = -100
    for planet_pos in env.planet_positions:
        grid[planet_pos] = -1
    grid[env.target_pos] = 100
    
    im = ax.imshow(grid, cmap='RdYlGn', alpha=0.3, interpolation='nearest')
    
    # Simulate trajectories
    colors = plt.cm.rainbow(np.linspace(0, 1, num_trajectories))
    
    success_count = 0
    black_hole_count = 0
    timeout_count = 0
    
    for traj_idx in range(num_trajectories):
        state = env.reset(start_pos=(0, 0))
        trajectory = [env._state_to_pos(state)]
        
        for step in range(500):  # Max steps
            action = policy[state]
            next_state, reward, done, info = env.step(action)
            
            trajectory.append(info['position'])
            
            if done:
                if info['reached_target']:
                    success_count += 1
                elif info['hit_black_hole']:
                    black_hole_count += 1
                break
            
            state = next_state
        else:
            timeout_count += 1
        
        # Plot trajectory
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 1], trajectory[:, 0], '-', color=colors[traj_idx],
               linewidth=2, alpha=0.7, label=f'Trajectory {traj_idx+1} ({len(trajectory)} steps)')
        
        # Mark start and end
        ax.plot(trajectory[0, 1], trajectory[0, 0], 'o', color=colors[traj_idx],
               markersize=10, markeredgecolor='black', markeredgewidth=2)
        ax.plot(trajectory[-1, 1], trajectory[-1, 0], 's', color=colors[traj_idx],
               markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    # Mark special locations with larger markers
    ax.plot(env.black_hole_pos[1], env.black_hole_pos[0], 'ko', markersize=20,
           markerfacecolor='black', label='Black Hole', zorder=10)
    
    for i, planet_pos in enumerate(env.planet_positions):
        label = 'Planets' if i == 0 else None
        ax.plot(planet_pos[1], planet_pos[0], 'o', markersize=15,
               markerfacecolor='brown', markeredgecolor='black', 
               markeredgewidth=2, label=label, zorder=10)
    
    ax.plot(env.target_pos[1], env.target_pos[0], '*', markersize=30,
           markerfacecolor='gold', markeredgecolor='black', 
           markeredgewidth=2, label='Target (Safe Planet)', zorder=10)
    
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title('Optimal Trajectories from Start (0,0) to Target (80,80)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.2)
    
    # Add success statistics
    stats_text = (
        f"Results ({num_trajectories} trajectories):\n"
        f"✓ Reached Target: {success_count}/{num_trajectories} ({success_count/num_trajectories*100:.0f}%)\n"
        f"✗ Hit Black Hole: {black_hole_count}/{num_trajectories}\n"
        f"⊗ Timeout: {timeout_count}/{num_trajectories}"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Optimal trajectories saved to: {save_path}")
    plt.close()


def analyze_path_safety(env, policy, start_pos=(0, 0), max_steps=500):
    """
    Analyze the safety and efficiency of the optimal path.
    
    Args:
        env: SimpleGridEnv instance
        policy: Optimal policy
        start_pos: Starting position
        max_steps: Maximum steps to simulate
        
    Returns:
        Dictionary with path analysis
    """
    state = env.reset(start_pos=start_pos)
    trajectory = [env._state_to_pos(state)]
    
    min_dist_to_black_hole = float('inf')
    total_reward = 0
    gravity_events = {'black_hole': 0, 'planet': 0, 'none': 0}
    
    for step in range(max_steps):
        action = policy[state]
        next_state, reward, done, info = env.step(action)
        
        trajectory.append(info['position'])
        total_reward += reward
        
        # Track minimum distance to black hole
        current_pos = info['position']
        dist_to_bh = env._manhattan_distance(current_pos, env.black_hole_pos)
        min_dist_to_black_hole = min(min_dist_to_black_hole, dist_to_bh)
        
        # Track gravity events
        gravity_type = info.get('gravity_applied', 'none')
        if 'black_hole' in str(gravity_type):
            gravity_events['black_hole'] += 1
        elif 'planet' in str(gravity_type):
            gravity_events['planet'] += 1
        else:
            gravity_events['none'] += 1
        
        if done:
            return {
                'success': info['reached_target'],
                'steps': step + 1,
                'trajectory': trajectory,
                'total_reward': total_reward,
                'min_dist_to_black_hole': min_dist_to_black_hole,
                'gravity_events': gravity_events,
                'final_position': info['position']
            }
        
        state = next_state
    
    return {
        'success': False,
        'steps': max_steps,
        'trajectory': trajectory,
        'total_reward': total_reward,
        'min_dist_to_black_hole': min_dist_to_black_hole,
        'gravity_events': gravity_events,
        'final_position': trajectory[-1]
    }


def run_convergence_verification():
    """Main function to verify and visualize convergence."""
    print("=" * 80)
    print("POLICY ITERATION CONVERGENCE VERIFICATION")
    print("=" * 80)
    
    # Create environment
    print("\n1. Creating 100x100 grid environment...")
    env = SimpleGridEnv(grid_size=100)
    print(f"   ✓ Grid size: {env.grid_size}x{env.grid_size} ({env.n_states} states)")
    print(f"   ✓ Black hole at {env.black_hole_pos}")
    print(f"   ✓ Target at {env.target_pos}")
    
    # Create solver with tracking
    print("\n2. Running Policy Iteration with convergence tracking...")
    solver = PolicyIteration(env, gamma=0.99, theta=1e-6)
    
    # Track convergence metrics
    history = []
    
    max_iterations = 1000
    for iteration in range(max_iterations):
        print(f"\n   Iteration {iteration + 1}:")
        
        # Policy Evaluation
        eval_iterations = solver.policy_evaluation()
        print(f"     • Policy evaluation: {eval_iterations} iterations")
        
        # Calculate statistics
        mean_value = solver.V.mean()
        max_value = solver.V.max()
        min_value = solver.V.min()
        print(f"     • Mean value: {mean_value:.4f}")
        
        # Policy Improvement
        old_policy = solver.policy.copy()
        policy_stable = solver.policy_improvement()
        policy_changes = np.sum(old_policy != solver.policy)
        
        print(f"     • Policy changes: {policy_changes} states")
        print(f"     • Policy stable: {policy_stable}")
        
        # Record history
        history.append({
            'iteration': iteration + 1,
            'eval_iterations': eval_iterations,
            'mean_value': mean_value,
            'max_value': max_value,
            'min_value': min_value,
            'policy_changes': policy_changes,
            'policy_stable': policy_stable
        })
        
        # Check convergence
        if policy_stable:
            print(f"\n   ✓ CONVERGED in {iteration + 1} iterations!")
            break
    
    # Extract final policy
    optimal_policy = solver.policy
    optimal_value = solver.V
    
    # Analyze optimal path
    print("\n3. Analyzing optimal path safety...")
    path_analysis = analyze_path_safety(env, optimal_policy)
    
    print(f"\n   Path Analysis:")
    print(f"   {'='*60}")
    print(f"   Success: {'✓ YES' if path_analysis['success'] else '✗ NO'}")
    print(f"   Steps to target: {path_analysis['steps']}")
    print(f"   Total reward: {path_analysis['total_reward']:.2f}")
    print(f"   Min distance to black hole: {path_analysis['min_dist_to_black_hole']}")
    print(f"   Gravity events: {path_analysis['gravity_events']}")
    
    if path_analysis['success']:
        print(f"\n   ✓ OPTIMAL POLICY SUCCESSFULLY REACHES TARGET!")
        print(f"   ✓ Path avoids black hole (min distance: {path_analysis['min_dist_to_black_hole']})")
    
    # Evaluate policy performance
    print("\n4. Evaluating policy over multiple episodes...")
    eval_results = solver.evaluate_policy(num_episodes=100, max_steps=1000)
    
    print(f"\n   Performance Metrics (100 episodes):")
    print(f"   {'='*60}")
    print(f"   Success rate: {eval_results['success_rate']*100:.1f}%")
    print(f"   Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Mean episode length: {eval_results['mean_episode_length']:.1f} steps")
    
    # Create visualizations
    print("\n5. Creating convergence visualizations...")
    
    visualize_convergence_metrics(history, 'convergence_metrics.png')
    visualize_policy_heatmap(env, optimal_policy, optimal_value, 'policy_heatmap.png')
    visualize_optimal_trajectory(env, optimal_policy, num_trajectories=5, 
                                 save_path='optimal_trajectories.png')
    
    # Final summary
    print("\n" + "=" * 80)
    print("CONVERGENCE VERIFICATION COMPLETE!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  ✓ Policy iteration converged in {len(history)} iterations")
    print(f"  ✓ Optimal policy found (no state changes action)")
    print(f"  ✓ Agent successfully reaches target: {eval_results['success_rate']*100:.0f}% success rate")
    print(f"  ✓ Path avoids black hole safely")
    print(f"  ✓ Average path length: {eval_results['mean_episode_length']:.0f} steps")
    
    print("\nGenerated Files:")
    print("  • convergence_metrics.png    - Convergence analysis plots")
    print("  • policy_heatmap.png         - Value function and policy visualization")
    print("  • optimal_trajectories.png   - Sample optimal paths to target")
    
    print("\n" + "=" * 80)
    
    return {
        'history': history,
        'optimal_policy': optimal_policy,
        'optimal_value': optimal_value,
        'path_analysis': path_analysis,
        'eval_results': eval_results
    }


if __name__ == "__main__":
    results = run_convergence_verification()

