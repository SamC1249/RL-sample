"""
Gymnasium wrappers for the gravitational environment.
Includes action space discretization for DQN compatibility.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Wrapper to convert hybrid Box action space to Discrete for DQN.
    
    Original action space: Box([0, 0], [3, 1]) - [direction, thrust]
    Wrapped action space: Discrete(n_directions * n_thrust_levels)
    """
    
    def __init__(self, env, n_thrust_levels: int = 5):
        super().__init__(env)
        
        self.n_directions = 4  # up, right, down, left
        self.n_thrust_levels = n_thrust_levels
        self.n_actions = self.n_directions * self.n_thrust_levels
        
        # Discretized thrust values
        self.thrust_values = np.linspace(0.0, 1.0, n_thrust_levels)
        
        # New action space is discrete
        self.action_space = spaces.Discrete(self.n_actions)
    
    def action(self, action: int) -> np.ndarray:
        """
        Convert discrete action to continuous [direction, thrust].
        
        Args:
            action: Integer in [0, n_actions)
            
        Returns:
            [direction, thrust]: Box action for the base environment
        """
        direction = action // self.n_thrust_levels
        thrust_idx = action % self.n_thrust_levels
        thrust = self.thrust_values[thrust_idx]
        
        return np.array([direction, thrust], dtype=np.float32)


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normalize observations to [-1, 1] range for better neural network training.
    
    Original: [x, y, vx, vy] where x,y in [0, 100], vx,vy in [-10, 10]
    Normalized: All values in approximately [-1, 1]
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get original observation space bounds
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        
        # New normalized observation space
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [-1, 1]"""
        # Linear scaling: (obs - low) / (high - low) * 2 - 1
        normalized = 2.0 * (obs - self.obs_low) / (self.obs_high - self.obs_low) - 1.0
        return normalized.astype(np.float32)


class RewardShaping(gym.RewardWrapper):
    """
    Optional: Apply reward shaping for faster learning.
    Adds distance-based reward to encourage progress toward target.
    """
    
    def __init__(self, env, distance_reward_scale: float = 0.01):
        super().__init__(env)
        self.distance_reward_scale = distance_reward_scale
        self.prev_distance = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_distance = info.get('distance_to_target', 100.0)
        return obs, info
    
    def reward(self, reward: float) -> float:
        """Add distance-based reward shaping"""
        # Get current distance to target
        info = self.env.unwrapped._get_info()
        current_distance = info.get('distance_to_target', 100.0)
        
        # Reward for getting closer to target
        if self.prev_distance is not None:
            distance_reward = (self.prev_distance - current_distance) * self.distance_reward_scale
            reward += distance_reward
        
        self.prev_distance = current_distance
        return reward


def make_wrapped_env(normalize_obs: bool = True, reward_shaping: bool = False, n_thrust_levels: int = 5):
    """
    Create wrapped environment for DQN training.
    
    Args:
        normalize_obs: Whether to normalize observations
        reward_shaping: Whether to add distance-based reward shaping
        n_thrust_levels: Number of discrete thrust levels
        
    Returns:
        Wrapped environment compatible with DQN
    """
    from gravitational_env import GravitationalDynamicsEnv
    
    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=1.0)
    
    # Discretize action space (required for DQN)
    env = DiscreteActionWrapper(env, n_thrust_levels=n_thrust_levels)
    
    # Optional: Normalize observations
    if normalize_obs:
        env = NormalizeObservation(env)
    
    # Optional: Add reward shaping
    if reward_shaping:
        env = RewardShaping(env, distance_reward_scale=0.01)
    
    return env


if __name__ == "__main__":
    # Test the wrapper
    print("="*80)
    print("Testing DiscreteActionWrapper")
    print("="*80)
    
    env = make_wrapped_env(normalize_obs=True, reward_shaping=False, n_thrust_levels=5)
    
    print(f"\nOriginal action space: Box([0, 0], [3, 1])")
    print(f"Wrapped action space: {env.action_space}")
    print(f"Total discrete actions: {env.action_space.n}")
    
    print(f"\nObservation space: {env.observation_space}")
    
    # Test episode
    obs, info = env.reset()
    print(f"\nInitial observation (normalized): {obs}")
    
    # Take random action
    action = env.action_space.sample()
    print(f"\nSample discrete action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After step - Reward: {reward:.4f}")
    print(f"New observation (normalized): {obs}")
    
    print("\n" + "="*80)
    print("Wrapper test successful!")
    print("="*80)

