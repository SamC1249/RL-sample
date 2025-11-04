"""
Quick test script to verify the environment works correctly
"""

import numpy as np
from astrophysics_env import AstrophysicsEnv


def test_environment_basic():
    """Test basic environment functionality"""
    print("Testing basic environment functionality...")

    # Create environment
    env = AstrophysicsEnv(grid_size=1000, max_fuel=500.0, max_steps=100, seed=42)

    # Test reset
    state, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Initial state shape: {state.shape}")
    print(f"  Initial position: {info['rocket_pos']}")
    print(f"  Initial fuel: {info['fuel']}")

    # Test action space
    print(f"\n✓ Action space: {env.action_space}")
    print(f"  Action space sample: {env.action_space.sample()}")

    # Test observation space
    print(f"\n✓ Observation space: {env.observation_space}")

    # Test step
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"\n✓ Step function works")
    print(f"  Action taken: {action}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")

    print("\n✓ All basic tests passed!")
    env.close()


def test_environment_episode():
    """Test a full episode"""
    print("\n" + "="*60)
    print("Testing full episode with random actions...")
    print("="*60)

    env = AstrophysicsEnv(grid_size=1000, max_fuel=500.0, max_steps=100, seed=42)
    state, info = env.reset()

    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 100:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1
        state = next_state

    print(f"✓ Episode completed")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final position: {info['rocket_pos']}")
    print(f"  Final velocity: {info['rocket_vel']}")
    print(f"  Remaining fuel: {info['fuel']:.2f}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")

    env.close()


def test_celestial_objects():
    """Test celestial objects generation"""
    print("\n" + "="*60)
    print("Testing celestial objects...")
    print("="*60)

    env = AstrophysicsEnv(grid_size=1000, max_fuel=500.0, seed=42)
    env.reset()

    print(f"✓ Celestial objects generated:")
    print(f"  Suns: {len(env.suns)}")
    print(f"  Black holes: {len(env.black_holes)}")
    print(f"  Landable planets: {len(env.landable_planets)}")
    print(f"  Non-landable planets: {len(env.non_landable_planets)}")
    print(f"  Asteroids: {len(env.asteroids)}")

    # Check objects are within bounds
    all_objects = (env.suns + env.black_holes + env.landable_planets +
                   env.non_landable_planets + env.asteroids)

    for obj in all_objects:
        assert 0 <= obj.x <= env.grid_size, f"Object x out of bounds: {obj.x}"
        assert 0 <= obj.y <= env.grid_size, f"Object y out of bounds: {obj.y}"

    print(f"✓ All objects within bounds")

    env.close()


def test_physics():
    """Test physics and gravity"""
    print("\n" + "="*60)
    print("Testing physics and gravity...")
    print("="*60)

    env = AstrophysicsEnv(grid_size=1000, max_fuel=500.0, seed=42)
    state, info = env.reset()

    initial_pos = info['rocket_pos'].copy()
    initial_vel = info['rocket_vel'].copy()

    # Apply thrust
    action = np.array([1.0, 0.0], dtype=np.float32)  # Full thrust in x-direction
    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"✓ Physics simulation:")
    print(f"  Initial position: {initial_pos}")
    print(f"  Initial velocity: {initial_vel}")
    print(f"  Action: thrust={action[0]:.2f}, angle={action[1]:.2f}")
    print(f"  New position: {info['rocket_pos']}")
    print(f"  New velocity: {info['rocket_vel']}")
    print(f"  Velocity changed: {not np.allclose(initial_vel, info['rocket_vel'])}")

    env.close()


def test_rewards():
    """Test reward structure"""
    print("\n" + "="*60)
    print("Testing reward structure...")
    print("="*60)

    env = AstrophysicsEnv(grid_size=1000, max_fuel=500.0, seed=42)
    state, info = env.reset()

    # Test action cost
    action = np.array([0.0, 0.0], dtype=np.float32)  # No thrust
    next_state, reward, _, _, info = env.step(action)
    print(f"✓ Base action reward (no thrust): {reward}")
    assert reward <= -1, "Base reward should be -1 or less"

    # Test fuel consumption
    initial_fuel = info['fuel']
    action = np.array([1.0, 0.0], dtype=np.float32)  # Full thrust
    next_state, reward, _, _, info = env.step(action)
    print(f"✓ Fuel consumed: {initial_fuel - info['fuel']:.2f}")
    assert info['fuel'] < initial_fuel, "Fuel should decrease with thrust"

    env.close()


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ASTROPHYSICS ENVIRONMENT TEST SUITE")
    print("="*60)

    try:
        test_environment_basic()
        test_environment_episode()
        test_celestial_objects()
        test_physics()
        test_rewards()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe environment is ready to use!")
        print("Try training an agent with: python train.py --algorithm ppo --episodes 100")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
