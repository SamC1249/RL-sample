"""
Quick start script for Gravitational Dynamics Environment
Demonstrates basic usage and quick training test.
"""

import numpy as np
from gravitational_env import GravitationalDynamicsEnv
from dqn_agent import DQNAgent, RandomAgent, HeuristicAgent
from train import train_agent
from evaluate import evaluate_agent, record_trajectory, visualize_trajectory


def test_environment():
    """Test basic environment functionality"""
    print("="*80)
    print("Testing Environment")
    print("="*80)

    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=0.1)

    print(f"\nEnvironment Configuration:")
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  Gravitational constant G: {env.G}")
    print(f"  Thrust reward coefficient k: {env.k}")

    print(f"\nCelestial Bodies:")
    for body in env.celestial_bodies:
        print(f"  {body.type:20s} - pos=({body.x:5.1f}, {body.y:5.1f}), "
              f"mass={body.mass:.1e}, reward={body.reward:6.1f}")
        if body.event_horizon:
            print(f"                           Event Horizon Radius: {body.event_horizon}")

    # Test episode
    print(f"\n{'='*80}")
    print("Running Test Episode")
    print("="*80)

    obs, info = env.reset()
    print(f"\nInitial state: {obs}")
    print(f"Initial info: {info}")

    # Take a few random steps
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {step + 1}:")
        print(f"  Action: direction={int(action[0])}, thrust={action[1]:.3f}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Position: ({obs[0]:.2f}, {obs[1]:.2f})")
        print(f"  Velocity: ({obs[2]:.2f}, {obs[3]:.2f})")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        total_reward += reward

        if terminated or truncated:
            print(f"\nEpisode ended after {step + 1} steps")
            break

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"{'='*80}\n")


def test_agents():
    """Test different agent types"""
    print("="*80)
    print("Testing Agents")
    print("="*80)

    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=0.1)

    # Test Random Agent
    print("\n1. Random Agent")
    print("-" * 40)
    random_agent = RandomAgent()
    state, _ = env.reset()

    for i in range(3):
        action = random_agent.select_action(state)
        print(f"  Action {i+1}: direction={int(action[0])}, thrust={action[1]:.3f}")

    # Test Heuristic Agent
    print("\n2. Heuristic Agent")
    print("-" * 40)
    heuristic_agent = HeuristicAgent(env)
    state, _ = env.reset()

    for i in range(3):
        action = heuristic_agent.select_action(state)
        print(f"  Action {i+1}: direction={int(action[0])}, thrust={action[1]:.3f}")
        state, _, _, _, _ = env.step(action)

    # Test DQN Agent
    print("\n3. DQN Agent (untrained)")
    print("-" * 40)
    dqn_agent = DQNAgent(
        state_dim=4,
        n_directions=4,
        n_thrust_levels=5,
        hidden_dims=[128, 128],
        epsilon_start=0.1  # Low epsilon for demonstration
    )
    state, _ = env.reset()

    for i in range(3):
        action = dqn_agent.select_action(state, training=False)
        print(f"  Action {i+1}: direction={int(action[0])}, thrust={action[1]:.3f}")

    print(f"\n{'='*80}\n")


def quick_training():
    """Run a quick training test"""
    print("="*80)
    print("Quick Training Test (100 episodes)")
    print("="*80)

    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=0.1)

    # Train DQN for a short time
    agent = DQNAgent(
        state_dim=4,
        n_directions=4,
        n_thrust_levels=5,
        hidden_dims=[64, 64],  # Smaller network for quick test
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.99,
        batch_size=32
    )

    print("\nTraining DQN agent for 100 episodes...")
    tracker = train_agent(
        agent=agent,
        env=env,
        n_episodes=100,
        max_steps=500,
        save_dir="checkpoints/quick_test",
        save_freq=50,
        eval_freq=20,
        verbose=True
    )

    # Quick evaluation
    print("\n" + "="*80)
    print("Quick Evaluation")
    print("="*80)

    metrics = evaluate_agent(agent, env, n_episodes=10, verbose=False)
    print(f"\nResults (10 episodes):")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  Avg Reward: {metrics['avg_reward']:.2f}")
    print(f"  Black Hole Rate: {metrics['black_hole_rate']:.2%}")

    # Record trajectory
    print("\nRecording sample trajectory...")
    positions, actions, rewards, success, black_hole = record_trajectory(
        agent, env, max_steps=500
    )

    visualize_trajectory(
        env, positions, actions, rewards, success, black_hole,
        save_path="quick_test_trajectory.png"
    )

    print(f"\n{'='*80}\n")


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("GRAVITATIONAL DYNAMICS RL ENVIRONMENT - QUICK START")
    print("="*80 + "\n")

    # Menu
    print("Select an option:")
    print("  1. Test Environment")
    print("  2. Test Agents")
    print("  3. Quick Training (100 episodes)")
    print("  4. Run All")
    print("  0. Exit")

    choice = input("\nEnter choice (0-4): ").strip()

    if choice == '1':
        test_environment()
    elif choice == '2':
        test_agents()
    elif choice == '3':
        quick_training()
    elif choice == '4':
        test_environment()
        test_agents()
        quick_training()
    elif choice == '0':
        print("Exiting...")
    else:
        print("Invalid choice. Running all tests...")
        test_environment()
        test_agents()
        quick_training()

    print("\n" + "="*80)
    print("DONE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # If running non-interactively, run all tests
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        print("\nRunning all tests non-interactively...\n")
        test_environment()
        test_agents()
        quick_training()
    else:
        main()
