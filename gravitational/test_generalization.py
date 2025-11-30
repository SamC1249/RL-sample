"""
Test Policy Generalization
Tests how well the trained policy generalizes to:
1. New random seeds (different obstacle/target configurations)
2. New planet locations
3. Different starting positions

This reveals if the agent learned a generalizable navigation strategy
or just memorized the training environment layout.
"""

import numpy as np
from stable_baselines3 import PPO
from gravitational_env import GravitationalDynamicsEnv, CelestialBody
import matplotlib.pyplot as plt
from typing import List, Tuple

class GeneralizableGravEnv(GravitationalDynamicsEnv):
    """Modified environment that can randomize celestial body positions"""
    
    def __init__(self, randomize_positions=False, seed=None, **kwargs):
        self.randomize_positions = randomize_positions
        self.custom_seed = seed
        super().__init__(**kwargs)
    
    def _init_celestial_bodies(self):
        """Initialize celestial bodies with optional randomization"""
        if self.custom_seed is not None:
            np.random.seed(self.custom_seed)
        
        if self.randomize_positions:
            # Randomize positions while keeping reasonable spacing
            # Black hole in center region (40-60, 40-60)
            bh_x = np.random.uniform(40, 60)
            bh_y = np.random.uniform(40, 60)
            
            self.black_hole = CelestialBody(
                x=bh_x, y=bh_y,
                mass=1e8,
                body_type='black_hole',
                reward=-100.0
            )
            self.celestial_bodies.append(self.black_hole)
            
            # Random planets (ensuring they're not too close to black hole or each other)
            planet_positions = []
            for _ in range(3):
                attempts = 0
                while attempts < 100:
                    px = np.random.uniform(20, 80)
                    py = np.random.uniform(20, 80)
                    
                    # Check distance from black hole
                    dist_to_bh = np.sqrt((px - bh_x)**2 + (py - bh_y)**2)
                    if dist_to_bh < 15:  # Too close to black hole
                        attempts += 1
                        continue
                    
                    # Check distance from other planets
                    too_close = False
                    for other_pos in planet_positions:
                        dist = np.sqrt((px - other_pos[0])**2 + (py - other_pos[1])**2)
                        if dist < 10:
                            too_close = True
                            break
                    
                    if not too_close:
                        planet_positions.append((px, py))
                        break
                    attempts += 1
            
            # Add planets
            for pos in planet_positions:
                planet = CelestialBody(
                    x=pos[0], y=pos[1],
                    mass=1e5,
                    body_type='normal_planet',
                    reward=-1.0
                )
                self.celestial_bodies.append(planet)
            
            # Target planet in far corner (avoiding black hole)
            if bh_x < 50:  # Black hole on left
                tx = np.random.uniform(75, 95)
            else:  # Black hole on right
                tx = np.random.uniform(5, 25)
            
            if bh_y < 50:  # Black hole below
                ty = np.random.uniform(75, 95)
            else:  # Black hole above
                ty = np.random.uniform(5, 25)
            
            self.target_planet = CelestialBody(
                x=tx, y=ty,
                mass=1e5,
                body_type='target_planet',
                reward=0.0
            )
            self.celestial_bodies.append(self.target_planet)
            
        else:
            # Use original fixed positions
            super()._init_celestial_bodies()


def test_generalization(model_path: str, n_test_configs: int = 10, n_episodes_per_config: int = 10):
    """
    Test policy generalization across different environment configurations.
    
    Args:
        model_path: Path to trained model
        n_test_configs: Number of different random configurations to test
        n_episodes_per_config: Episodes per configuration
    """
    
    print("="*80)
    print("POLICY GENERALIZATION TEST")
    print("="*80)
    print(f"\nLoading model: {model_path}")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure to provide the path WITHOUT .zip extension")
        return
    
    print("✓ Model loaded\n")
    
    # Test 1: Original Environment (Training Configuration)
    print("\n" + "="*80)
    print("TEST 1: Original Environment (What it trained on)")
    print("="*80)
    
    env_original = GeneralizableGravEnv(randomize_positions=False)
    results_original = run_test_episodes(model, env_original, n_episodes_per_config)
    
    print_results("Original Config", results_original)
    
    # Test 2: Same seed, different random realizations
    print("\n" + "="*80)
    print("TEST 2: Randomized Configurations (Different planet layouts)")
    print("="*80)
    
    all_random_results = []
    
    for config_idx in range(n_test_configs):
        seed = 100 + config_idx  # Different seeds
        env_random = GeneralizableGravEnv(randomize_positions=True, seed=seed)
        
        print(f"\n  Config {config_idx+1}/{n_test_configs} (seed={seed}):")
        print(f"    Black hole: ({env_random.black_hole.x:.1f}, {env_random.black_hole.y:.1f})")
        print(f"    Target: ({env_random.target_planet.x:.1f}, {env_random.target_planet.y:.1f})")
        
        results = run_test_episodes(model, env_random, n_episodes_per_config, verbose=False)
        all_random_results.append(results)
        
        print(f"    Success: {results['success_rate']*100:.1f}% | "
              f"Black Hole: {results['black_hole_rate']*100:.1f}% | "
              f"Avg Reward: {results['avg_reward']:.2f}")
    
    # Aggregate random results
    random_aggregate = aggregate_results(all_random_results)
    
    # Test 3: Different starting positions (original layout)
    print("\n" + "="*80)
    print("TEST 3: Different Starting Positions (Original layout)")
    print("="*80)
    
    env_original_2 = GeneralizableGravEnv(randomize_positions=False)
    start_positions = [
        (10, 10),   # Original
        (10, 90),   # Top-left
        (90, 10),   # Bottom-right
        (50, 10),   # Bottom-center
        (20, 50),   # Left-center
    ]
    
    start_pos_results = []
    for start_pos in start_positions:
        results = run_test_episodes_from_position(model, env_original_2, start_pos, 
                                                   n_episodes_per_config, verbose=False)
        start_pos_results.append(results)
        print(f"  Start {start_pos}: Success {results['success_rate']*100:.1f}% | "
              f"Avg Reward: {results['avg_reward']:.2f}")
    
    start_pos_aggregate = aggregate_results(start_pos_results)
    
    # Final Summary
    print("\n" + "="*80)
    print("GENERALIZATION SUMMARY")
    print("="*80)
    
    print("\n📊 Performance Comparison:")
    print(f"\n1. Original Training Config:")
    print(f"   Success Rate:    {results_original['success_rate']*100:>6.1f}%")
    print(f"   Black Hole Rate: {results_original['black_hole_rate']*100:>6.1f}%")
    print(f"   Avg Reward:      {results_original['avg_reward']:>7.2f}")
    
    print(f"\n2. Random Configurations (Avg across {n_test_configs} configs):")
    print(f"   Success Rate:    {random_aggregate['success_rate']*100:>6.1f}%")
    print(f"   Black Hole Rate: {random_aggregate['black_hole_rate']*100:>6.1f}%")
    print(f"   Avg Reward:      {random_aggregate['avg_reward']:>7.2f}")
    
    print(f"\n3. Different Start Positions (Original layout):")
    print(f"   Success Rate:    {start_pos_aggregate['success_rate']*100:>6.1f}%")
    print(f"   Black Hole Rate: {start_pos_aggregate['black_hole_rate']*100:>6.1f}%")
    print(f"   Avg Reward:      {start_pos_aggregate['avg_reward']:>7.2f}")
    
    # Generalization score
    print("\n📈 Generalization Analysis:")
    
    baseline_success = results_original['success_rate']
    random_success = random_aggregate['success_rate']
    start_success = start_pos_aggregate['success_rate']
    
    if baseline_success > 0:
        random_ratio = random_success / baseline_success
        start_ratio = start_success / baseline_success
    else:
        random_ratio = 0
        start_ratio = 0
    
    print(f"\n   Random Config Performance: {random_ratio*100:.1f}% of original")
    print(f"   Different Start Performance: {start_ratio*100:.1f}% of original")
    
    if random_ratio > 0.7:
        print("\n   ✅ GOOD: Policy generalizes well to new configurations!")
        print("      Agent learned robust navigation strategy")
    elif random_ratio > 0.4:
        print("\n   ⚠️  MODERATE: Some generalization, but performance drops")
        print("      Agent partially learned spatial reasoning")
    else:
        print("\n   ❌ POOR: Policy doesn't generalize well")
        print("      Agent likely memorized specific layout")
        print("      Consider training on randomized environments")
    
    print("\n" + "="*80)
    
    return {
        'original': results_original,
        'random_configs': all_random_results,
        'random_aggregate': random_aggregate,
        'start_positions': start_pos_results,
        'start_aggregate': start_pos_aggregate
    }


def run_test_episodes(model, env, n_episodes: int, verbose: bool = True):
    """Run test episodes and collect statistics"""
    
    successes = 0
    black_holes = 0
    timeouts = 0
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done:
                if info.get('terminal_reason') == 'success':
                    successes += 1
                elif info.get('terminal_reason') == 'black_hole':
                    black_holes += 1
                break
            
            if truncated:
                timeouts += 1
                break
        
        rewards.append(episode_reward)
        lengths.append(step + 1)
    
    return {
        'success_rate': successes / n_episodes,
        'black_hole_rate': black_holes / n_episodes,
        'timeout_rate': timeouts / n_episodes,
        'avg_reward': np.mean(rewards),
        'avg_length': np.mean(lengths),
        'n_episodes': n_episodes
    }


def run_test_episodes_from_position(model, env, start_pos: Tuple[float, float], 
                                    n_episodes: int, verbose: bool = True):
    """Run test episodes from a specific starting position"""
    
    successes = 0
    black_holes = 0
    timeouts = 0
    rewards = []
    lengths = []
    
    for ep in range(n_episodes):
        # Reset and manually set starting position
        env.reset()
        env.agent_pos = np.array(start_pos, dtype=np.float32)
        env.agent_vel = np.array([0.0, 0.0], dtype=np.float32)
        state = env._get_observation()
        
        episode_reward = 0
        
        for step in range(1000):
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done:
                if info.get('terminal_reason') == 'success':
                    successes += 1
                elif info.get('terminal_reason') == 'black_hole':
                    black_holes += 1
                break
            
            if truncated:
                timeouts += 1
                break
        
        rewards.append(episode_reward)
        lengths.append(step + 1)
    
    return {
        'success_rate': successes / n_episodes,
        'black_hole_rate': black_holes / n_episodes,
        'timeout_rate': timeouts / n_episodes,
        'avg_reward': np.mean(rewards),
        'avg_length': np.mean(lengths),
        'n_episodes': n_episodes
    }


def aggregate_results(results_list: List[dict]) -> dict:
    """Aggregate multiple test results"""
    return {
        'success_rate': np.mean([r['success_rate'] for r in results_list]),
        'black_hole_rate': np.mean([r['black_hole_rate'] for r in results_list]),
        'timeout_rate': np.mean([r['timeout_rate'] for r in results_list]),
        'avg_reward': np.mean([r['avg_reward'] for r in results_list]),
        'avg_length': np.mean([r['avg_length'] for r in results_list]),
    }


def print_results(config_name: str, results: dict):
    """Print test results"""
    print(f"\n  Results for {config_name}:")
    print(f"    Success Rate:    {results['success_rate']*100:>6.1f}%")
    print(f"    Black Hole Rate: {results['black_hole_rate']*100:>6.1f}%")
    print(f"    Timeout Rate:    {results['timeout_rate']*100:>6.1f}%")
    print(f"    Avg Reward:      {results['avg_reward']:>7.2f}")
    print(f"    Avg Length:      {results['avg_length']:>7.1f} steps")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_generalization.py <model_path> [n_configs] [n_episodes]")
        print("\nExample:")
        print("  python test_generalization.py checkpoints/sb3_ppo_quick/final_model")
        print("  python test_generalization.py checkpoints/sb3_ppo_quick/interrupted_model 5 10")
        sys.exit(1)
    
    model_path = sys.argv[1]
    n_configs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    test_generalization(model_path, n_configs, n_episodes)

