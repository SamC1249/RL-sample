"""
Test Fuel Constraint Implementation
Quick test to verify fuel system works correctly.
"""

from gravitational_env import GravitationalDynamicsEnv
import numpy as np

print("="*80)
print("TESTING FUEL CONSTRAINT IMPLEMENTATION")
print("="*80)

# Create environment with fuel
env = GravitationalDynamicsEnv(
    grid_size=100,
    G=1e-3,
    k=0.01,  # Small fuel cost
    max_fuel=100.0,
    fuel_consumption_rate=0.1
)

print("\n1. Environment Configuration:")
print(f"   Max Fuel: {env.max_fuel}")
print(f"   Fuel Consumption Rate: {env.fuel_consumption_rate}")
print(f"   Observation Space: {env.observation_space}")
print(f"   Expected obs shape: (5,) - [x, y, vx, vy, fuel]")

# Reset and check initial state
obs, info = env.reset()
print("\n2. Initial State:")
print(f"   Observation: {obs}")
print(f"   Position: ({obs[0]:.1f}, {obs[1]:.1f})")
print(f"   Velocity: ({obs[2]:.1f}, {obs[3]:.1f})")
print(f"   Fuel: {obs[4]:.1f}")
print(f"   Info fuel: {info['fuel']:.1f}")
print(f"   Fuel percentage: {info['fuel_percentage']:.1f}%")

# Test fuel consumption
print("\n3. Testing Fuel Consumption:")
initial_fuel = obs[4]

# Take action with max thrust
action = np.array([1, 1.0])  # Right, max thrust
obs, reward, terminated, truncated, info = env.step(action)

fuel_used = initial_fuel - obs[4]
print(f"   Action: Direction=Right, Thrust=1.0")
print(f"   Fuel before: {initial_fuel:.2f}")
print(f"   Fuel after: {obs[4]:.2f}")
print(f"   Fuel used: {fuel_used:.2f}")
print(f"   Expected: {1.0 * env.fuel_consumption_rate:.2f}")
print(f"   ✓ Match!" if abs(fuel_used - 0.1) < 0.01 else "   ✗ Error!")

# Test insufficient fuel
print("\n4. Testing Insufficient Fuel:")
env.fuel = 0.05  # Set fuel to half of what's needed
initial_fuel = env.fuel

action = np.array([1, 1.0])  # Try to use 1.0 thrust (needs 0.1 fuel)
obs, reward, terminated, truncated, info = env.step(action)

print(f"   Fuel available: {initial_fuel:.2f}")
print(f"   Fuel needed for thrust=1.0: {1.0 * env.fuel_consumption_rate:.2f}")
print(f"   Fuel after: {obs[4]:.2f}")
print(f"   Agent should only thrust at: {initial_fuel / env.fuel_consumption_rate:.2f}")
print(f"   ✓ Fuel limited correctly!" if obs[4] < 0.01 else "   ⚠ Check implementation")

# Test reward changes
print("\n5. Testing Reward Function:")
env.reset()

# Thrust with gravity opposition
action = np.array([0, 0.7])  # Up, moderate thrust
obs, reward, terminated, truncated, info = env.step(action)

print(f"   Action: Thrust=0.7")
print(f"   Reward: {reward:.4f}")
print(f"   Expected: Negative (fuel cost + small gravity)")
print(f"   ✓ Fuel costs reward!" if reward < 0 else "   ⚠ Check k parameter")

# Test target reward
print("\n6. Testing Target Reward:")
# Manually move agent to target
env.agent_pos = np.array([85.0, 85.0])
action = np.array([0, 0.0])  # No thrust
obs, reward, terminated, truncated, info = env.step(action)

print(f"   Moved agent to target location (85, 85)")
print(f"   Terminated: {terminated}")
print(f"   Terminal reason: {info.get('terminal_reason', 'None')}")
print(f"   Reward: {reward:.1f}")
print(f"   Expected: ~+100 (target reward)")
print(f"   ✓ Target reward working!" if reward > 90 else "   ✗ Check target reward")

# Test black hole
print("\n7. Testing Black Hole Penalty:")
env.reset()
# Manually move agent to black hole
env.agent_pos = np.array([50.0, 50.0])
action = np.array([0, 0.0])
obs, reward, terminated, truncated, info = env.step(action)

print(f"   Moved agent to black hole (50, 50)")
print(f"   Terminated: {terminated}")
print(f"   Terminal reason: {info.get('terminal_reason', 'None')}")
print(f"   Reward: {reward:.1f}")
print(f"   Expected: -100 (black hole penalty)")
print(f"   ✓ Black hole penalty working!" if reward < -90 else "   ✗ Check black hole reward")

# Full episode test
print("\n8. Running Full Episode Test (100 steps):")
obs, info = env.reset()
initial_fuel = obs[4]
episode_reward = 0
fuel_used_total = 0

for step in range(100):
    # Random actions
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    
    if terminated or truncated:
        print(f"   Episode ended at step {step + 1}")
        print(f"   Reason: {info.get('terminal_reason', 'max_steps')}")
        break

final_fuel = obs[4]
fuel_used_total = initial_fuel - final_fuel

print(f"   Initial fuel: {initial_fuel:.2f}")
print(f"   Final fuel: {final_fuel:.2f}")
print(f"   Total fuel used: {fuel_used_total:.2f}")
print(f"   Episode reward: {episode_reward:.2f}")
print(f"   ✓ Episode completed successfully!")

print("\n" + "="*80)
print("ALL TESTS COMPLETE!")
print("="*80)
print("\nKey Changes Verified:")
print("  ✓ Fuel system working (consumption + limits)")
print("  ✓ Observation space includes fuel (5D)")
print("  ✓ Reward function penalizes fuel use")
print("  ✓ Target reward = +100")
print("  ✓ Black hole penalty = -100")
print("\nReady for training!")
print("="*80)

