"""
Example usage of the Astrophysics Environment
Demonstrates basic interaction with the environment
"""

import numpy as np
from astrophysics_env import AstrophysicsEnv


def random_agent_example():
    """
    Example 1: Random agent
    Demonstrates basic environment usage
    """
    print("="*60)
    print("Example 1: Random Agent")
    print("="*60)

    # Create environment
    env = AstrophysicsEnv(
        grid_size=1000,
        max_fuel=500.0,
        max_steps=200,
        seed=42
    )

    # Run 3 episodes
    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        print("-" * 40)

        state, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Random action
            action = env.action_space.sample()

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1
            state = next_state

        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Position: ({info['rocket_pos'][0]:.1f}, {info['rocket_pos'][1]:.1f})")
        print(f"Fuel Remaining: {info['fuel']:.2f}")

    env.close()


def smart_agent_example():
    """
    Example 2: Simple heuristic agent
    Attempts to navigate toward the nearest landable planet
    """
    print("\n" + "="*60)
    print("Example 2: Simple Heuristic Agent")
    print("="*60)

    env = AstrophysicsEnv(
        grid_size=1000,
        max_fuel=500.0,
        max_steps=300,
        seed=42
    )

    state, info = env.reset()
    total_reward = 0
    steps = 0
    done = False

    print(f"Starting position: {info['rocket_pos']}")
    print(f"Number of landable planets: {len(env.landable_planets)}")

    # Find nearest landable planet
    nearest_planet = min(env.landable_planets,
                        key=lambda p: p.distance_to(info['rocket_pos'][0],
                                                     info['rocket_pos'][1]))
    target_pos = np.array([nearest_planet.x, nearest_planet.y])
    print(f"Target planet at: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")

    while not done:
        # Simple heuristic: thrust toward target
        direction_to_target = target_pos - info['rocket_pos']
        distance_to_target = np.linalg.norm(direction_to_target)

        if distance_to_target > 0:
            # Calculate angle to target
            angle = np.arctan2(direction_to_target[1], direction_to_target[0])
            angle = angle % (2 * np.pi)

            # Adjust thrust based on distance
            if distance_to_target > 100:
                thrust = 0.8  # High thrust when far
            elif distance_to_target > 50:
                thrust = 0.5  # Medium thrust
            else:
                thrust = 0.2  # Low thrust when close

            action = np.array([thrust, angle], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1
        state = next_state

        # Print progress every 50 steps
        if steps % 50 == 0:
            current_dist = np.linalg.norm(target_pos - info['rocket_pos'])
            print(f"Step {steps}: Distance to target = {current_dist:.1f}, "
                  f"Fuel = {info['fuel']:.1f}")

    print(f"\nFinal Results:")
    print(f"Steps: {steps}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Position: ({info['rocket_pos'][0]:.1f}, {info['rocket_pos'][1]:.1f})")
    print(f"Distance to Target: {np.linalg.norm(target_pos - info['rocket_pos']):.1f}")
    print(f"Fuel Remaining: {info['fuel']:.2f}")

    if terminated and total_reward >= -steps:
        print("✓ Successfully landed on planet!")
    else:
        print("✗ Failed to land")

    env.close()


def analyze_environment():
    """
    Example 3: Analyze the environment
    Shows information about celestial objects and their properties
    """
    print("\n" + "="*60)
    print("Example 3: Environment Analysis")
    print("="*60)

    env = AstrophysicsEnv(grid_size=1000, max_fuel=500.0, seed=42)
    state, info = env.reset()

    print(f"\nEnvironment Configuration:")
    print(f"  Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"  Max Fuel: {env.max_fuel}")
    print(f"  Max Steps: {env.max_steps}")
    print(f"  Earth Position: ({env.earth_pos[0]:.1f}, {env.earth_pos[1]:.1f})")

    print(f"\nCelestial Objects:")
    print(f"  Suns: {len(env.suns)}")
    for i, sun in enumerate(env.suns):
        print(f"    Sun {i+1}: pos=({sun.x:.1f}, {sun.y:.1f}), "
              f"influence={sun.influence_radius:.1f}")

    print(f"  Black Holes: {len(env.black_holes)}")
    for i, bh in enumerate(env.black_holes):
        print(f"    Black Hole {i+1}: pos=({bh.x:.1f}, {bh.y:.1f}), "
              f"influence={bh.influence_radius:.1f}, "
              f"event_horizon={bh.event_horizon:.1f}")

    print(f"  Landable Planets: {len(env.landable_planets)}")
    print(f"  Non-landable Planets: {len(env.non_landable_planets)}")
    print(f"  Asteroids: {len(env.asteroids)}")

    print(f"\nState Space:")
    print(f"  Dimensions: {env.observation_space.shape[0]}")
    print(f"  Low bounds: {env.observation_space.low}")
    print(f"  High bounds: {env.observation_space.high}")

    print(f"\nAction Space:")
    print(f"  Type: Continuous")
    print(f"  Dimensions: {env.action_space.shape[0]}")
    print(f"  Low bounds: {env.action_space.low}")
    print(f"  High bounds: {env.action_space.high}")
    print(f"  Meaning: [thrust_magnitude (0-1), thrust_angle (0-2π)]")

    print(f"\nInitial State:")
    print(f"  Position: ({state[0]:.1f}, {state[1]:.1f})")
    print(f"  Velocity: ({state[2]:.3f}, {state[3]:.3f})")
    print(f"  Fuel: {state[4]:.1f}")
    print(f"  Nearest Sun: dist={state[5]:.1f}, angle={state[6]:.3f}")
    print(f"  Nearest Black Hole: dist={state[7]:.1f}, angle={state[8]:.3f}")
    print(f"  Nearest Planet: dist={state[9]:.1f}, angle={state[10]:.3f}")

    env.close()


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("ASTROPHYSICS ENVIRONMENT - EXAMPLE USAGE")
    print("="*60)

    # Run examples
    random_agent_example()
    smart_agent_example()
    analyze_environment()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the environment: python test_environment.py")
    print("2. Train an agent: python train.py --algorithm ppo --episodes 100")
    print("3. Evaluate trained agent: python evaluate.py --algorithm ppo --episodes 10 --visualize")


if __name__ == "__main__":
    main()
