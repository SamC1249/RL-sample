"""
Quick test script for trained DQN agent
Tests the most recent checkpoint and shows results.
"""

import numpy as np
from gravitational_env import GravitationalDynamicsEnv
from dqn_agent import DQNAgent
import os
import glob

def find_latest_checkpoint(checkpoint_dir="checkpoints/dqn"):
    """Find the most recent checkpoint file"""
    checkpoints = glob.glob(f"{checkpoint_dir}/checkpoint_ep*.pt")
    if not checkpoints:
        return None
    # Sort by episode number
    checkpoints.sort(key=lambda x: int(x.split('_ep')[-1].split('.')[0]))
    return checkpoints[-1]

def test_agent(checkpoint_path=None, n_episodes=10):
    """Test the trained agent"""
    
    print("=" * 60)
    print("TESTING TRAINED DQN AGENT")
    print("=" * 60)
    
    # Create environment
    env = GravitationalDynamicsEnv()
    
    # Create agent
    agent = DQNAgent(
        state_dim=4,
        n_directions=4,
        n_thrust_levels=5,
        hidden_dims=[128, 128]
    )
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("ERROR: No checkpoints found!")
            return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)
    print(f"✓ Checkpoint loaded successfully")
    print(f"  Epsilon: {agent.epsilon:.4f}")
    
    # Run test episodes
    print(f"\nRunning {n_episodes} test episodes...")
    print("-" * 60)
    
    total_reward = 0
    success_count = 0
    black_hole_count = 0
    timeout_count = 0
    episode_lengths = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            # Select action (training=False disables exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            
            if terminated:
                if info.get('terminal_reason') == 'success':
                    success_count += 1
                    outcome = "✓ SUCCESS"
                elif info.get('terminal_reason') == 'black_hole':
                    black_hole_count += 1
                    outcome = "✗ BLACK HOLE"
                break
            
            if truncated:
                timeout_count += 1
                outcome = "⊗ TIMEOUT"
                break
        
        total_reward += episode_reward
        episode_lengths.append(step + 1)
        
        print(f"Episode {episode+1:2d}: {outcome:15s} | "
              f"Reward: {episode_reward:7.2f} | Steps: {step+1:4d}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Episodes:        {n_episodes}")
    print(f"Success Rate:          {success_count}/{n_episodes} ({success_count/n_episodes*100:.1f}%)")
    print(f"Black Hole Deaths:     {black_hole_count}/{n_episodes} ({black_hole_count/n_episodes*100:.1f}%)")
    print(f"Timeouts:              {timeout_count}/{n_episodes} ({timeout_count/n_episodes*100:.1f}%)")
    print(f"Average Reward:        {total_reward/n_episodes:.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
    print("=" * 60)
    
    # Show single detailed episode
    print("\nShowing detailed trajectory for one episode...")
    state, _ = env.reset()
    trajectory = [info['agent_position'].copy()]
    
    for step in range(1000):
        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        trajectory.append(info['agent_position'].copy())
        
        if terminated or truncated:
            break
    
    print(f"\nTrajectory ({len(trajectory)} steps):")
    print(f"  Start:  {trajectory[0]}")
    if len(trajectory) > 10:
        print(f"  ...     ({len(trajectory)-10} intermediate steps)")
        for i in range(-5, 0):
            print(f"  Step {len(trajectory)+i}: {trajectory[i]}")
    print(f"  End:    {trajectory[-1]}")
    
    if info.get('terminal_reason') == 'success':
        print(f"\n✓ Agent successfully reached target at (85, 85)!")
    elif info.get('terminal_reason') == 'black_hole':
        print(f"\n✗ Agent hit black hole at (50, 50)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import sys
    
    checkpoint = None
    n_episodes = 10
    
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    if len(sys.argv) > 2:
        n_episodes = int(sys.argv[2])
    
    test_agent(checkpoint, n_episodes)

