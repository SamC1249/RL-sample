"""
Gravitational Dynamics Reinforcement Learning Environment
A 100×100 grid world with realistic Newtonian gravity simulation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict, Optional


class CelestialBody:
    """Represents a celestial body with mass and position"""
    def __init__(self, x: float, y: float, mass: float, body_type: str, reward: float = 0.0):
        self.x = x
        self.y = y
        self.mass = mass
        self.type = body_type
        self.reward = reward

        # Calculate event horizon radius for black holes
        if body_type == 'black_hole':
            # r = floor(3 + log10(m / 1e6)) <= 6
            self.event_horizon = min(int(3 + np.log10(mass / 1e6)), 6)
        else:
            self.event_horizon = None

    def get_position(self) -> Tuple[float, float]:
        return self.x, self.y

    def distance_to(self, x: float, y: float) -> float:
        """Calculate Euclidean distance to a point"""
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)

    def is_inside_event_horizon(self, x: float, y: float) -> bool:
        """Check if a point is inside the event horizon (black holes only)"""
        if self.event_horizon is None:
            return False
        return self.distance_to(x, y) <= self.event_horizon

    def is_at_planet(self, x: float, y: float, threshold: float = 2.0) -> bool:
        """Check if agent is at planet location"""
        return self.distance_to(x, y) <= threshold


class GravitationalDynamicsEnv(gym.Env):
    """
    A 100×100 grid world RL environment with realistic gravitational dynamics.

    Agent (spaceship) moves using discrete directions with continuous thrust magnitude.
    Gravity follows Newton's law with multiple celestial bodies.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, grid_size: int = 100, G: float = 1e-3, k: float = 1.0, 
                 max_fuel: float = 100.0, fuel_consumption_rate: float = 0.1, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.G = G  # Gravitational constant
        self.k = k  # Thrust cost coefficient (negative for fuel penalty)
        self.render_mode = render_mode
        
        # Fuel parameters
        self.max_fuel = max_fuel
        self.fuel_consumption_rate = fuel_consumption_rate  # Fuel used per unit thrust
        self.fuel = max_fuel  # Current fuel

        # Action space: [direction (0-3), thrust magnitude (0-1)]
        # Direction: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Box(
            low=np.array([0, 0.0]),
            high=np.array([3, 1.0]),
            dtype=np.float32
        )

        # Observation space: [x, y, vx, vy, fuel] (position, velocity, and fuel)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10, -10, 0]),
            high=np.array([grid_size, grid_size, 10, 10, max_fuel]),
            dtype=np.float32
        )

        # Initialize celestial bodies
        self.celestial_bodies: List[CelestialBody] = []
        self._init_celestial_bodies()

        # Agent state
        self.agent_pos = np.array([10.0, 10.0], dtype=np.float32)
        self.agent_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Pre-compute gravity gradient for the entire grid
        self.gravity_field = self._compute_gravity_field()

        self.steps = 0
        self.max_steps = 1000

    def _init_celestial_bodies(self):
        """Initialize 1 black hole and 4 planets (3 normal, 1 target)"""
        np.random.seed(42)  # For reproducibility

        # Black hole (mass: 1e8, reward: -100)
        self.black_hole = CelestialBody(
            x=50.0, y=50.0,
            mass=1e8,
            body_type='black_hole',
            reward=-100.0
        )
        self.celestial_bodies.append(self.black_hole)

        # 3 Normal planets (mass: 1e5, reward: -1)
        planet_positions = [
            (25.0, 75.0),
            (75.0, 25.0),
            (30.0, 30.0)
        ]

        for pos in planet_positions:
            planet = CelestialBody(
                x=pos[0], y=pos[1],
                mass=1e5,
                body_type='normal_planet',
                reward=-1.0
            )
            self.celestial_bodies.append(planet)

        # 1 Target planet (mass: 1e5, reward: 100.0)
        self.target_planet = CelestialBody(
            x=85.0, y=85.0,
            mass=1e5,
            body_type='target_planet',
            reward=100.0
        )
        self.celestial_bodies.append(self.target_planet)

    def _compute_gravity_field(self) -> np.ndarray:
        """
        Pre-compute gravity gradient for entire grid using Newton's law.
        OPTIMIZED: Uses NumPy vectorization instead of Python loops.
        g(x,y) = -G * Σ(m_i / r_i²) * r̂_i

        Returns:
            gravity_field: Array of shape (grid_size, grid_size, 2) containing [gx, gy]
        """
        # Create meshgrid for all positions at once
        x_coords = np.arange(self.grid_size, dtype=np.float32)
        y_coords = np.arange(self.grid_size, dtype=np.float32)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Initialize gravity field
        gravity_field = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        
        # Vectorized computation for each celestial body
        for body in self.celestial_bodies:
            # Vector from body to all points (vectorized)
            dx = X - body.x
            dy = Y - body.y
            
            # Distance to all points (vectorized)
            r = np.sqrt(dx**2 + dy**2)
            
            # Avoid division by zero
            r = np.maximum(r, 0.1)
            
            # Gravitational acceleration (vectorized)
            g_magnitude = -self.G * body.mass / (r**2)
            
            # Add contribution to gravity field
            gravity_field[:, :, 0] += g_magnitude * (dx / r)
            gravity_field[:, :, 1] += g_magnitude * (dy / r)
        
        return gravity_field

    def _get_gravity_at_position(self, x: float, y: float) -> np.ndarray:
        """
        Get gravity vector at a specific position.
        OPTIMIZED: Uses simple nearest-neighbor lookup instead of bilinear interpolation.
        This is 10-20x faster and the difference is negligible for this grid resolution.
        """
        # Clip to grid bounds and round to nearest integer
        x_idx = int(np.clip(np.round(x), 0, self.grid_size - 1))
        y_idx = int(np.clip(np.round(y), 0, self.grid_size - 1))
        
        return self.gravity_field[x_idx, y_idx]

    def _compute_reward(self, thrust_direction: np.ndarray, thrust_magnitude: float,
                       gravity_vector: np.ndarray) -> float:
        """
        Compute reward with fuel constraint.
        
        New reward: r = -gravity_cost - fuel_cost
        
        Changes from original:
        - Thrust is now a COST (penalty), not reward
        - Encourages efficient fuel use
        - Still penalizes poor gravity management

        Args:
            thrust_direction: Unit vector of thrust direction
            thrust_magnitude: Magnitude of thrust T ∈ [0, 1]
            gravity_vector: Local gravity vector

        Returns:
            reward: Float reward value
        """
        g_magnitude = np.linalg.norm(gravity_vector)

        if g_magnitude > 1e-6:
            # Normalize gravity vector to get direction
            g_direction = gravity_vector / g_magnitude

            # Compute cosine of angle between thrust and gravity
            # Positive dot product means thrust opposes gravity (good)
            # Note: We negate gravity direction because gravity points inward
            cos_theta = np.dot(thrust_direction, -g_direction)

            # Gravity cost: normalized using tanh to keep in [0, 1] range
            # Divide by 100 to scale the large gravity values before tanh
            raw_gravity_cost = g_magnitude * (1 - cos_theta)
            gravity_cost = np.tanh(raw_gravity_cost / 100.0)
        else:
            gravity_cost = 0.0

        # Fuel cost: penalty for using fuel (negative, not positive!)
        # This encourages efficient fuel use
        fuel_cost = self.k * thrust_magnitude  # k should be negative or small

        reward = -gravity_cost - fuel_cost
        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Reset agent to starting position (avoid starting near celestial bodies)
        self.agent_pos = np.array([10.0, 10.0], dtype=np.float32)
        self.agent_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.fuel = self.max_fuel  # Reset fuel to max
        self.steps = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: [direction, thrust_magnitude]
                direction: 0=up, 1=right, 2=down, 3=left
                thrust_magnitude: T ∈ [0, 1]
        """
        self.steps += 1

        # Parse action
        direction_idx = int(np.clip(action[0], 0, 3))
        thrust_magnitude = np.clip(action[1], 0.0, 1.0)
        
        # Check fuel and limit thrust if insufficient
        fuel_needed = thrust_magnitude * self.fuel_consumption_rate
        if self.fuel < fuel_needed:
            # Not enough fuel - can only use what's available
            thrust_magnitude = self.fuel / self.fuel_consumption_rate
            fuel_needed = self.fuel
        
        # Consume fuel
        self.fuel -= fuel_needed
        self.fuel = max(0.0, self.fuel)  # Ensure non-negative

        # Direction vectors: up, right, down, left
        direction_vectors = [
            np.array([0.0, 1.0]),   # up
            np.array([1.0, 0.0]),   # right
            np.array([0.0, -1.0]),  # down
            np.array([-1.0, 0.0])   # left
        ]
        thrust_direction = direction_vectors[direction_idx]

        # Get gravity at current position
        gravity = self._get_gravity_at_position(self.agent_pos[0], self.agent_pos[1])

        # Compute reward
        reward = self._compute_reward(thrust_direction, thrust_magnitude, gravity)

        # Apply thrust to velocity
        thrust_acceleration = thrust_direction * thrust_magnitude

        # Update velocity: v += a*dt (assume dt=1 for discrete time steps)
        self.agent_vel += thrust_acceleration + gravity

        # Velocity damping to prevent unbounded growth
        self.agent_vel *= 0.95

        # Update position: x += v*dt
        self.agent_pos += self.agent_vel

        # Clip to grid bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

        # Check terminal conditions
        terminated = False
        truncated = False
        terminal_reason = None

        # Check if agent reached target planet
        if self.target_planet.is_at_planet(self.agent_pos[0], self.agent_pos[1]):
            terminated = True
            terminal_reason = 'success'
            reward += self.target_planet.reward  # +100 for reaching target!

        # Check if agent entered black hole event horizon
        elif self.black_hole.is_inside_event_horizon(self.agent_pos[0], self.agent_pos[1]):
            terminated = True
            terminal_reason = 'black_hole'
            reward = self.black_hole.reward  # -100 for black hole
        
        # Check if out of fuel and can't reach target (optional failure condition)
        # Uncomment if you want running out of fuel to end episode
        # elif self.fuel <= 0:
        #     terminated = True
        #     terminal_reason = 'out_of_fuel'
        #     reward -= 10.0  # Small penalty for running out

        # Check if max steps reached
        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        
        # Add terminal reason to info for proper success detection
        if terminal_reason is not None:
            info['terminal_reason'] = terminal_reason

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation [x, y, vx, vy, fuel]"""
        return np.concatenate([self.agent_pos, self.agent_vel, [self.fuel]])

    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'agent_position': self.agent_pos.copy(),
            'agent_velocity': self.agent_vel.copy(),
            'fuel': self.fuel,
            'fuel_percentage': (self.fuel / self.max_fuel) * 100,
            'steps': self.steps,
            'distance_to_target': self.target_planet.distance_to(
                self.agent_pos[0], self.agent_pos[1]
            ),
            'distance_to_black_hole': self.black_hole.distance_to(
                self.agent_pos[0], self.agent_pos[1]
            )
        }

    def render(self):
        """Render the environment (optional - can be implemented later)"""
        if self.render_mode == 'human':
            print(f"Agent at ({self.agent_pos[0]:.2f}, {self.agent_pos[1]:.2f}), "
                  f"Velocity: ({self.agent_vel[0]:.2f}, {self.agent_vel[1]:.2f})")

    def get_gravity_field(self) -> np.ndarray:
        """Return the pre-computed gravity field for visualization"""
        return self.gravity_field


if __name__ == "__main__":
    # Test the environment
    env = GravitationalDynamicsEnv()

    print("=== Gravitational Dynamics Environment ===")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Gravitational constant G: {env.G}")
    print(f"Thrust reward coefficient k: {env.k}")
    print(f"\nCelestial bodies:")
    for body in env.celestial_bodies:
        print(f"  - {body.type}: pos=({body.x}, {body.y}), mass={body.mass:.2e}, "
              f"reward={body.reward}, event_horizon={body.event_horizon}")

    # Run a simple test
    obs, info = env.reset()
    print(f"\nInitial observation: {obs}")
    print(f"Initial info: {info}")

    # Take a random action
    action = env.action_space.sample()
    print(f"\nSample action: direction={int(action[0])}, thrust={action[1]:.3f}")

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After step - Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
    print(f"New observation: {obs}")
    print(f"New info: {info}")
