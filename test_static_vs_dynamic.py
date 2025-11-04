"""
Demonstration: Static vs Dynamic Environment

This script shows the difference between static and dynamic environment modes.
"""

import numpy as np
from astrophysics_env import AstrophysicsEnv


def test_static_environment():
    """Test static environment - objects stay in same positions"""
    print("="*60)
    print("STATIC ENVIRONMENT TEST")
    print("="*60)
    print("In static mode, celestial objects remain in the same positions")
    print("across multiple episodes, making it easier for agents to learn.\n")

    env = AstrophysicsEnv(
        grid_size=1000,
        max_fuel=500.0,
        seed=42,
        static_environment=True  # Static mode
    )

    # Record object positions from first reset
    state1, _ = env.reset()
    first_episode_objects = {
        'landable_planets': [(p.x, p.y) for p in env.landable_planets[:3]],
        'black_holes': [(bh.x, bh.y) for bh in env.black_holes],
        'suns': [(s.x, s.y) for s in env.suns[:3]]
    }

    print("First episode - First 3 landable planets:")
    for i, (x, y) in enumerate(first_episode_objects['landable_planets']):
        print(f"  Planet {i+1}: ({x:.1f}, {y:.1f})")

    # Reset again (new episode)
    state2, _ = env.reset()
    second_episode_objects = {
        'landable_planets': [(p.x, p.y) for p in env.landable_planets[:3]],
        'black_holes': [(bh.x, bh.y) for bh in env.black_holes],
        'suns': [(s.x, s.y) for s in env.suns[:3]]
    }

    print("\nSecond episode - First 3 landable planets:")
    for i, (x, y) in enumerate(second_episode_objects['landable_planets']):
        print(f"  Planet {i+1}: ({x:.1f}, {y:.1f})")

    # Check if positions are identical
    positions_match = (first_episode_objects == second_episode_objects)

    print(f"\n✓ Positions match: {positions_match}")
    print("Static environment: Same configuration every episode!")

    env.close()


def test_dynamic_environment():
    """Test dynamic environment - objects change positions each episode"""
    print("\n" + "="*60)
    print("DYNAMIC ENVIRONMENT TEST")
    print("="*60)
    print("In dynamic mode, celestial objects are randomly repositioned")
    print("on each reset, creating different challenges each episode.\n")

    env = AstrophysicsEnv(
        grid_size=1000,
        max_fuel=500.0,
        seed=42,
        static_environment=False  # Dynamic mode
    )

    # Record object positions from first reset
    state1, _ = env.reset()
    first_episode_objects = {
        'landable_planets': [(p.x, p.y) for p in env.landable_planets[:3]],
        'black_holes': [(bh.x, bh.y) for bh in env.black_holes],
        'suns': [(s.x, s.y) for s in env.suns[:3]]
    }

    print("First episode - First 3 landable planets:")
    for i, (x, y) in enumerate(first_episode_objects['landable_planets']):
        print(f"  Planet {i+1}: ({x:.1f}, {y:.1f})")

    # Reset again (new episode)
    state2, _ = env.reset()
    second_episode_objects = {
        'landable_planets': [(p.x, p.y) for p in env.landable_planets[:3]],
        'black_holes': [(bh.x, bh.y) for bh in env.black_holes],
        'suns': [(s.x, s.y) for s in env.suns[:3]]
    }

    print("\nSecond episode - First 3 landable planets:")
    for i, (x, y) in enumerate(second_episode_objects['landable_planets']):
        print(f"  Planet {i+1}: ({x:.1f}, {y:.1f})")

    # Check if positions are different
    positions_match = (first_episode_objects == second_episode_objects)

    print(f"\n✓ Positions match: {positions_match}")
    print("Dynamic environment: Different configuration every episode!")

    env.close()


def test_seed_consistency():
    """Test that same seed produces same initial configuration"""
    print("\n" + "="*60)
    print("SEED CONSISTENCY TEST")
    print("="*60)
    print("Using the same seed should produce the same initial configuration,")
    print("even in static mode.\n")

    # Create two environments with same seed
    env1 = AstrophysicsEnv(grid_size=1000, seed=123, static_environment=True)
    env2 = AstrophysicsEnv(grid_size=1000, seed=123, static_environment=True)

    env1.reset()
    env2.reset()

    # Compare planet positions
    env1_planets = [(p.x, p.y) for p in env1.landable_planets[:3]]
    env2_planets = [(p.x, p.y) for p in env2.landable_planets[:3]]

    print("Environment 1 - First 3 planets:")
    for i, (x, y) in enumerate(env1_planets):
        print(f"  Planet {i+1}: ({x:.1f}, {y:.1f})")

    print("\nEnvironment 2 (same seed) - First 3 planets:")
    for i, (x, y) in enumerate(env2_planets):
        print(f"  Planet {i+1}: ({x:.1f}, {y:.1f})")

    match = (env1_planets == env2_planets)
    print(f"\n✓ Configurations match: {match}")
    print("Same seed produces reproducible environments!")

    env1.close()
    env2.close()


def test_observation_consistency():
    """Test that observations are consistent in static mode"""
    print("\n" + "="*60)
    print("OBSERVATION CONSISTENCY TEST")
    print("="*60)
    print("In static mode, the same state should produce the same observation")
    print("across episodes (except for rocket state which always resets).\n")

    env = AstrophysicsEnv(grid_size=1000, seed=42, static_environment=True)

    # Get initial observations from two episodes
    obs1, _ = env.reset()
    obs2, _ = env.reset()

    print("First episode initial state:")
    print(f"  Position: ({obs1[0]:.1f}, {obs1[1]:.1f})")
    print(f"  Nearest planet dist: {obs1[9]:.1f}")
    print(f"  Nearest planet angle: {obs1[10]:.3f}")

    print("\nSecond episode initial state:")
    print(f"  Position: ({obs2[0]:.1f}, {obs2[1]:.1f})")
    print(f"  Nearest planet dist: {obs2[9]:.1f}")
    print(f"  Nearest planet angle: {obs2[10]:.3f}")

    # Check if object-related observations are the same
    objects_match = np.allclose(obs1[5:], obs2[5:])  # Skip rocket state, check object info
    print(f"\n✓ Object observations match: {objects_match}")
    print("Static mode provides consistent environment for learning!")

    env.close()


def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print(" "*10 + "STATIC vs DYNAMIC ENVIRONMENT DEMONSTRATION")
    print("="*70)

    test_static_environment()
    test_dynamic_environment()
    test_seed_consistency()
    test_observation_consistency()

    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR TRAINING:")
    print("="*70)
    print("""
1. START WITH STATIC (default):
   - Use: python train.py --algorithm ppo --episodes 500
   - Agent learns optimal paths for one specific configuration
   - Faster convergence, easier debugging
   - Good for initial training and algorithm development

2. PROGRESS TO DYNAMIC (advanced):
   - Use: python train.py --algorithm ppo --episodes 500 --dynamic
   - Agent must generalize across different configurations
   - Slower learning but more robust policies
   - Use after agent masters static environment

3. CURRICULUM LEARNING (recommended):
   - Train on static environment first (500 episodes)
   - Fine-tune on dynamic environment (500 more episodes)
   - Best of both worlds: fast initial learning + generalization
    """)

    print("="*70)


if __name__ == "__main__":
    main()
