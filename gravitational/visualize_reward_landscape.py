"""
Visualize Reward Landscape and Proposed Improvements
Shows current vs improved reward functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gravitational_env import GravitationalDynamicsEnv
import seaborn as sns

sns.set_style("whitegrid")


def visualize_current_reward_landscape():
    """Visualize current reward function across the grid"""
    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Sample rewards at each grid position (assuming agent thrusts away from BH)
    grid_size = 100
    reward_map = np.zeros((grid_size, grid_size))
    gravity_magnitude_map = np.zeros((grid_size, grid_size))
    
    black_hole_pos = np.array([50.0, 50.0])
    
    for i in range(grid_size):
        for j in range(grid_size):
            pos = np.array([float(i), float(j)])
            gravity = env._get_gravity_at_position(i, j)
            g_mag = np.linalg.norm(gravity)
            gravity_magnitude_map[i, j] = g_mag
            
            # Assume agent thrusts away from black hole with moderate thrust
            if g_mag > 1e-6:
                thrust_direction = -gravity / g_mag  # Opposite of gravity
            else:
                thrust_direction = np.array([1.0, 0.0])
            
            thrust_magnitude = 0.7
            reward = env._compute_reward(thrust_direction, thrust_magnitude, gravity)
            reward_map[i, j] = reward
    
    # 1. Current Reward Landscape
    ax1 = axes[0, 0]
    im1 = ax1.imshow(reward_map.T, origin='lower', cmap='RdYlGn', 
                     extent=[0, 100, 0, 100], vmin=-1, vmax=1)
    
    # Add celestial bodies
    circle_bh = patches.Circle((50, 50), 6, color='black', alpha=0.8, label='Black Hole')
    ax1.add_patch(circle_bh)
    ax1.plot(85, 85, 'g*', markersize=20, label='Target')
    ax1.plot(10, 10, 'ro', markersize=10, label='Start')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Current Reward Landscape\n(Thrust away from BH, T=0.7)')
    ax1.legend(loc='upper left')
    plt.colorbar(im1, ax=ax1, label='Reward')
    
    # 2. Gravity Magnitude
    ax2 = axes[0, 1]
    im2 = ax2.imshow(gravity_magnitude_map.T, origin='lower', cmap='YlOrRd',
                     extent=[0, 100, 0, 100])
    
    circle_bh2 = patches.Circle((50, 50), 6, color='black', alpha=0.8)
    ax2.add_patch(circle_bh2)
    ax2.plot(85, 85, 'g*', markersize=20)
    ax2.plot(10, 10, 'ro', markersize=10)
    
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Gravity Magnitude ||g||')
    plt.colorbar(im2, ax=ax2, label='||g||')
    
    # 3. Distance to Target
    ax3 = axes[1, 0]
    target_pos = np.array([85.0, 85.0])
    distance_map = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            pos = np.array([float(i), float(j)])
            distance_map[i, j] = np.linalg.norm(pos - target_pos)
    
    im3 = ax3.imshow(distance_map.T, origin='lower', cmap='plasma',
                     extent=[0, 100, 0, 100])
    
    circle_bh3 = patches.Circle((50, 50), 6, color='black', alpha=0.8)
    ax3.add_patch(circle_bh3)
    ax3.plot(85, 85, 'g*', markersize=20)
    ax3.plot(10, 10, 'ro', markersize=10)
    
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Distance to Target\n(NOT in current reward!)')
    plt.colorbar(im3, ax=ax3, label='Distance')
    
    # 4. Proposed Improved Reward
    ax4 = axes[1, 1]
    improved_reward_map = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            pos = np.array([float(i), float(j)])
            
            # Gravity danger
            gravity = env._get_gravity_at_position(i, j)
            g_mag = np.linalg.norm(gravity)
            dist_to_bh = np.linalg.norm(pos - black_hole_pos)
            
            if dist_to_bh < 20:
                gravity_danger = -0.5 * g_mag * (20 - dist_to_bh) / 20
            else:
                gravity_danger = -0.1 * g_mag
            
            # Distance to target
            dist_to_target = np.linalg.norm(pos - target_pos)
            distance_reward = -0.02 * dist_to_target
            
            # Fuel cost
            fuel_cost = -0.01 * 0.7  # Assume T=0.7
            
            improved_reward = fuel_cost + gravity_danger + distance_reward
            improved_reward_map[i, j] = improved_reward
    
    im4 = ax4.imshow(improved_reward_map.T, origin='lower', cmap='RdYlGn',
                     extent=[0, 100, 0, 100])
    
    circle_bh4 = patches.Circle((50, 50), 6, color='black', alpha=0.8)
    ax4.add_patch(circle_bh4)
    ax4.plot(85, 85, 'g*', markersize=20)
    ax4.plot(10, 10, 'ro', markersize=10)
    
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    ax4.set_title('Proposed Improved Reward\n(Distance + Gravity + Fuel)')
    plt.colorbar(im4, ax=ax4, label='Reward')
    
    plt.suptitle('Reward Landscape Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reward_landscape_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: reward_landscape_analysis.png")
    plt.show()


def plot_reward_comparison():
    """Compare current vs improved reward along key paths"""
    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=1.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Path 1: Start to Target (optimal path)
    start = np.array([10.0, 10.0])
    target = np.array([85.0, 85.0])
    n_points = 100
    path_optimal = np.linspace(start, target, n_points)
    
    current_rewards = []
    improved_rewards = []
    gravity_costs = []
    distances = []
    
    for pos in path_optimal:
        gravity = env._get_gravity_at_position(pos[0], pos[1])
        g_mag = np.linalg.norm(gravity)
        
        # Current reward
        if g_mag > 1e-6:
            thrust_dir = -gravity / g_mag
        else:
            thrust_dir = np.array([1.0, 0.0])
        
        thrust_mag = 0.7
        current_reward = env._compute_reward(thrust_dir, thrust_mag, gravity)
        current_rewards.append(current_reward)
        gravity_costs.append(-g_mag)
        
        # Improved reward
        dist_to_target = np.linalg.norm(pos - target)
        dist_to_bh = np.linalg.norm(pos - np.array([50.0, 50.0]))
        
        if dist_to_bh < 20:
            grav_danger = -0.5 * g_mag * (20 - dist_to_bh) / 20
        else:
            grav_danger = -0.1 * g_mag
        
        distance_reward = -0.02 * dist_to_target
        fuel_cost = -0.01 * thrust_mag
        improved_reward = fuel_cost + grav_danger + distance_reward
        improved_rewards.append(improved_reward)
        distances.append(dist_to_target)
    
    # Plot 1: Rewards along optimal path
    ax1 = axes[0, 0]
    ax1.plot(current_rewards, label='Current Reward', linewidth=2, alpha=0.7)
    ax1.plot(improved_rewards, label='Improved Reward', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Step along path')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Along Optimal Path (Start → Target)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Distance to target
    ax2 = axes[0, 1]
    ax2.plot(distances, linewidth=2, color='green', alpha=0.7)
    ax2.set_xlabel('Step along path')
    ax2.set_ylabel('Distance to Target')
    ax2.set_title('Distance Decreases Along Optimal Path')
    ax2.grid(alpha=0.3)
    
    # Path 2: Start to wrong corner (what agent learned)
    wrong_corner = np.array([99.0, 0.0])
    path_wrong = np.linspace(start, wrong_corner, n_points)
    
    current_rewards_wrong = []
    improved_rewards_wrong = []
    
    for pos in path_wrong:
        gravity = env._get_gravity_at_position(pos[0], pos[1])
        g_mag = np.linalg.norm(gravity)
        
        if g_mag > 1e-6:
            thrust_dir = -gravity / g_mag
        else:
            thrust_dir = np.array([1.0, 0.0])
        
        thrust_mag = 0.7
        current_reward = env._compute_reward(thrust_dir, thrust_mag, gravity)
        current_rewards_wrong.append(current_reward)
        
        # Improved
        dist_to_target = np.linalg.norm(pos - target)
        dist_to_bh = np.linalg.norm(pos - np.array([50.0, 50.0]))
        
        if dist_to_bh < 20:
            grav_danger = -0.5 * g_mag * (20 - dist_to_bh) / 20
        else:
            grav_danger = -0.1 * g_mag
        
        distance_reward = -0.02 * dist_to_target
        fuel_cost = -0.01 * thrust_mag
        improved_reward = fuel_cost + grav_danger + distance_reward
        improved_rewards_wrong.append(improved_reward)
    
    # Plot 3: Comparison of paths (current reward)
    ax3 = axes[1, 0]
    ax3.plot(current_rewards, label='Optimal Path (→ Target)', 
            linewidth=2, color='green', alpha=0.7)
    ax3.plot(current_rewards_wrong, label='Wrong Path (→ Corner)', 
            linewidth=2, color='red', alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Current Reward')
    ax3.set_title('Current Reward: Both Paths Look Similar! 😞')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 4: Comparison of paths (improved reward)
    ax4 = axes[1, 1]
    ax4.plot(improved_rewards, label='Optimal Path (→ Target)', 
            linewidth=2, color='green', alpha=0.7)
    ax4.plot(improved_rewards_wrong, label='Wrong Path (→ Corner)', 
            linewidth=2, color='red', alpha=0.7)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Improved Reward')
    ax4.set_title('Improved Reward: Clear Preference for Target! 😊')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.suptitle('Why Current Reward Doesn\'t Work', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reward_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: reward_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("="*80)
    print("REWARD LANDSCAPE VISUALIZATION")
    print("="*80)
    print("\nGenerating visualizations...")
    print("This will show WHY the agent goes to the wrong corner!\n")
    
    visualize_current_reward_landscape()
    plot_reward_comparison()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Current reward is UNIFORM away from black hole")
    print("   → No gradient toward target!")
    print("\n2. Both paths (target vs wrong corner) get similar rewards")
    print("   → Agent can't tell which is better!")
    print("\n3. Improved reward creates clear gradient toward target")
    print("   → Agent will learn correct path!")
    print("="*80)

