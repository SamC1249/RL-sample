"""
Custom Gym Environment for Astrophysics Rocket Navigation
A 1000x1000 grid environment with gravity dynamics, celestial objects, and fuel management.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Tuple, List, Dict, Optional


class CelestialObject:
    """Base class for celestial objects"""
    def __init__(self, x: float, y: float, radius: float, object_type: str):
        self.x = x
        self.y = y
        self.radius = radius
        self.type = object_type

    def get_position(self) -> Tuple[float, float]:
        return self.x, self.y

    def distance_to(self, x: float, y: float) -> float:
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)


class Sun(CelestialObject):
    """Sun with strong gravitational influence (50x50 grid)"""
    def __init__(self, x: float, y: float):
        super().__init__(x, y, radius=5.0, object_type='sun')
        self.gravity_strength = 0.5
        self.influence_radius = 50.0


class BlackHole(CelestialObject):
    """Black hole with very strong gravitational influence (100x100 grid)"""
    def __init__(self, x: float, y: float):
        super().__init__(x, y, radius=3.0, object_type='black_hole')
        self.gravity_strength = 2.0
        self.influence_radius = 100.0
        self.event_horizon = 10.0  # Point of no return


class Planet(CelestialObject):
    """Planet that can be landed on"""
    def __init__(self, x: float, y: float, can_land: bool):
        super().__init__(x, y, radius=3.0, object_type='planet')
        self.can_land = can_land


class Asteroid(CelestialObject):
    """Asteroid obstacle"""
    def __init__(self, x: float, y: float):
        super().__init__(x, y, radius=2.0, object_type='asteroid')


class AstrophysicsEnv(gym.Env):
    """
    Custom Gym environment for rocket navigation in space with gravity and celestial objects.

    State Space:
        - Rocket position: (x, y)
        - Rocket velocity: (vx, vy)
        - Fuel remaining: fuel
        - Nearby object information (distance and angle to nearest objects)

    Action Space:
        - Continuous: [thrust_magnitude (0-1), thrust_angle (0-2π)]

    Rewards:
        - Each action: -1
        - Hitting asteroid: -2
        - Within black hole influence but not event horizon: -2 per step
        - Reaching black hole center or hitting sun: -100
        - Landing on valid planet: 0 (terminal)
        - Out of fuel and drifting: -2 per step

    Environment Modes:
        - static_environment=True (default): Celestial objects remain in the same positions
          across episodes. This makes learning much easier as the agent can learn optimal
          paths for a specific configuration. Recommended for initial training.

        - static_environment=False: Celestial objects are randomly repositioned on each reset.
          This makes the task much harder but results in more generalizable policies.
          Recommended after the agent has mastered a static configuration.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, grid_size: int = 1000, max_fuel: float = 500.0,
                 max_steps: int = 2000, render_mode: Optional[str] = None,
                 seed: Optional[int] = None, static_environment: bool = True):
        super().__init__()

        self.grid_size = grid_size
        self.max_fuel = max_fuel
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.seed_value = seed
        self.static_environment = static_environment
        self.environment_initialized = False

        # Earth position (center of grid)
        self.earth_pos = np.array([grid_size / 2, grid_size / 2], dtype=np.float32)

        # Define action and observation spaces
        # Action: [thrust_magnitude (0-1), thrust_angle (0-2π)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 2 * np.pi]),
            dtype=np.float32
        )

        # Observation: [x, y, vx, vy, fuel, nearest_sun_dist, nearest_sun_angle,
        #              nearest_blackhole_dist, nearest_blackhole_angle,
        #              nearest_planet_dist, nearest_planet_angle]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -50, -50, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([grid_size, grid_size, 50, 50, max_fuel,
                          grid_size, 2*np.pi, grid_size, 2*np.pi, grid_size, 2*np.pi],
                         dtype=np.float32)
        )

        # Initialize celestial objects
        self.suns: List[Sun] = []
        self.black_holes: List[BlackHole] = []
        self.landable_planets: List[Planet] = []
        self.non_landable_planets: List[Planet] = []
        self.asteroids: List[Asteroid] = []

        # Rocket state
        self.rocket_pos = None
        self.rocket_vel = None
        self.fuel = None
        self.steps = 0

        # For rendering
        self.fig = None
        self.ax = None

        # Initialize environment
        self.reset(seed=seed)

    def _generate_celestial_objects(self):
        """Generate all celestial objects with random positions"""
        rng = np.random.RandomState(self.seed_value)

        # Clear existing objects
        self.suns = []
        self.black_holes = []
        self.landable_planets = []
        self.non_landable_planets = []
        self.asteroids = []

        # Generate 2-3 black holes
        num_black_holes = rng.randint(2, 4)
        for _ in range(num_black_holes):
            pos = self._generate_valid_position(rng, min_distance_from_earth=200)
            self.black_holes.append(BlackHole(pos[0], pos[1]))

        # Generate 5-8 suns
        num_suns = rng.randint(5, 9)
        for _ in range(num_suns):
            pos = self._generate_valid_position(rng, min_distance_from_earth=100)
            self.suns.append(Sun(pos[0], pos[1]))

        # Generate 10 landable planets
        for _ in range(10):
            pos = self._generate_valid_position(rng, min_distance_from_earth=50)
            self.landable_planets.append(Planet(pos[0], pos[1], can_land=True))

        # Generate 120 non-landable planets
        for _ in range(120):
            pos = self._generate_valid_position(rng, min_distance_from_earth=50)
            self.non_landable_planets.append(Planet(pos[0], pos[1], can_land=False))

        # Generate 50-100 asteroids
        num_asteroids = rng.randint(50, 101)
        for _ in range(num_asteroids):
            pos = self._generate_valid_position(rng, min_distance_from_earth=30)
            self.asteroids.append(Asteroid(pos[0], pos[1]))

    def _generate_valid_position(self, rng, min_distance_from_earth: float = 50) -> np.ndarray:
        """Generate a random position that's not too close to Earth or other objects"""
        max_attempts = 100
        for _ in range(max_attempts):
            pos = rng.uniform(50, self.grid_size - 50, size=2)

            # Check distance from Earth
            if np.linalg.norm(pos - self.earth_pos) < min_distance_from_earth:
                continue

            # Check distance from existing objects (avoid overlaps)
            valid = True
            for obj_list in [self.suns, self.black_holes, self.landable_planets,
                            self.non_landable_planets]:
                for obj in obj_list:
                    if obj.distance_to(pos[0], pos[1]) < 30:  # Minimum separation
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                return pos

        # Fallback: return random position if we can't find a valid one
        return rng.uniform(50, self.grid_size - 50, size=2)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state

        Args:
            seed: Random seed for environment generation
            options: Additional options (can contain 'static_environment' to override)

        Returns:
            observation, info tuple
        """
        if seed is not None:
            self.seed_value = seed
            self.environment_initialized = False  # Force regeneration with new seed
            np.random.seed(seed)

        # Handle options
        if options is not None and 'static_environment' in options:
            self.static_environment = options['static_environment']

        # Generate celestial objects only if:
        # 1. First time (not initialized), OR
        # 2. Dynamic environment mode (static_environment=False)
        if not self.environment_initialized or not self.static_environment:
            self._generate_celestial_objects()
            self.environment_initialized = True

        # Reset rocket state (start at Earth)
        self.rocket_pos = self.earth_pos.copy()
        self.rocket_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.fuel = self.max_fuel
        self.steps = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _compute_gravitational_force(self) -> np.ndarray:
        """Compute total gravitational force from all celestial objects"""
        total_force = np.array([0.0, 0.0], dtype=np.float32)

        # Gravity from suns
        for sun in self.suns:
            dist = sun.distance_to(self.rocket_pos[0], self.rocket_pos[1])
            if dist < sun.influence_radius and dist > 0:
                direction = np.array([sun.x - self.rocket_pos[0],
                                     sun.y - self.rocket_pos[1]])
                direction = direction / dist  # Normalize
                # Gravity force inversely proportional to distance squared
                force_magnitude = sun.gravity_strength / (dist ** 2) * 100
                total_force += direction * force_magnitude

        # Gravity from black holes (much stronger)
        for bh in self.black_holes:
            dist = bh.distance_to(self.rocket_pos[0], self.rocket_pos[1])
            if dist < bh.influence_radius and dist > 0:
                direction = np.array([bh.x - self.rocket_pos[0],
                                     bh.y - self.rocket_pos[1]])
                direction = direction / dist  # Normalize
                # Stronger gravity for black holes
                force_magnitude = bh.gravity_strength / (dist ** 2) * 100
                total_force += direction * force_magnitude

        return total_force

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step"""
        self.steps += 1
        reward = -1.0  # Base cost for each action
        terminated = False
        truncated = False

        # Parse action
        thrust_magnitude = np.clip(action[0], 0.0, 1.0)
        thrust_angle = action[1] % (2 * np.pi)

        # Apply thrust if fuel is available
        thrust_force = np.array([0.0, 0.0], dtype=np.float32)
        if self.fuel > 0:
            fuel_consumption = thrust_magnitude * 0.5  # Fuel consumption rate
            self.fuel = max(0, self.fuel - fuel_consumption)

            # Thrust force
            max_thrust = 2.0
            thrust_force = max_thrust * thrust_magnitude * np.array([
                np.cos(thrust_angle),
                np.sin(thrust_angle)
            ])
        else:
            # Out of fuel penalty
            reward -= 2.0

        # Compute gravitational forces
        gravity_force = self._compute_gravitational_force()

        # Update velocity (F = ma, assume mass = 1)
        total_force = thrust_force + gravity_force
        self.rocket_vel += total_force * 0.1  # Time step = 0.1

        # Limit velocity
        speed = np.linalg.norm(self.rocket_vel)
        if speed > 50:
            self.rocket_vel = self.rocket_vel / speed * 50

        # Update position
        self.rocket_pos += self.rocket_vel * 0.1

        # Check boundaries (wrap around or bounce)
        if self.rocket_pos[0] < 0 or self.rocket_pos[0] > self.grid_size:
            self.rocket_vel[0] *= -0.5  # Bounce with energy loss
            self.rocket_pos[0] = np.clip(self.rocket_pos[0], 0, self.grid_size)
        if self.rocket_pos[1] < 0 or self.rocket_pos[1] > self.grid_size:
            self.rocket_vel[1] *= -0.5
            self.rocket_pos[1] = np.clip(self.rocket_pos[1], 0, self.grid_size)

        # Check collisions and special conditions
        # Check black holes
        for bh in self.black_holes:
            dist = bh.distance_to(self.rocket_pos[0], self.rocket_pos[1])
            if dist < bh.radius:  # Hit center of black hole
                reward = -100
                terminated = True
                break
            elif dist < bh.event_horizon:  # Within event horizon
                reward -= 2.0

        # Check suns
        for sun in self.suns:
            dist = sun.distance_to(self.rocket_pos[0], self.rocket_pos[1])
            if dist < sun.radius:  # Hit sun
                reward = -100
                terminated = True
                break

        # Check asteroids
        for asteroid in self.asteroids:
            dist = asteroid.distance_to(self.rocket_pos[0], self.rocket_pos[1])
            if dist < asteroid.radius:
                reward -= 2.0
                # Remove asteroid after collision
                self.asteroids.remove(asteroid)
                break

        # Check landable planets (terminal states with reward 0)
        if not terminated:
            for planet in self.landable_planets:
                dist = planet.distance_to(self.rocket_pos[0], self.rocket_pos[1])
                if dist < planet.radius * 1.5:  # Landing zone
                    reward = 0
                    terminated = True
                    break

        # Check max steps
        if self.steps >= self.max_steps:
            truncated = True

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Find nearest sun
        nearest_sun_dist = self.grid_size
        nearest_sun_angle = 0
        for sun in self.suns:
            dist = sun.distance_to(self.rocket_pos[0], self.rocket_pos[1])
            if dist < nearest_sun_dist:
                nearest_sun_dist = dist
                dx = sun.x - self.rocket_pos[0]
                dy = sun.y - self.rocket_pos[1]
                nearest_sun_angle = np.arctan2(dy, dx) % (2 * np.pi)

        # Find nearest black hole
        nearest_bh_dist = self.grid_size
        nearest_bh_angle = 0
        for bh in self.black_holes:
            dist = bh.distance_to(self.rocket_pos[0], self.rocket_pos[1])
            if dist < nearest_bh_dist:
                nearest_bh_dist = dist
                dx = bh.x - self.rocket_pos[0]
                dy = bh.y - self.rocket_pos[1]
                nearest_bh_angle = np.arctan2(dy, dx) % (2 * np.pi)

        # Find nearest landable planet
        nearest_planet_dist = self.grid_size
        nearest_planet_angle = 0
        for planet in self.landable_planets:
            dist = planet.distance_to(self.rocket_pos[0], self.rocket_pos[1])
            if dist < nearest_planet_dist:
                nearest_planet_dist = dist
                dx = planet.x - self.rocket_pos[0]
                dy = planet.y - self.rocket_pos[1]
                nearest_planet_angle = np.arctan2(dy, dx) % (2 * np.pi)

        obs = np.array([
            self.rocket_pos[0],
            self.rocket_pos[1],
            self.rocket_vel[0],
            self.rocket_vel[1],
            self.fuel,
            nearest_sun_dist,
            nearest_sun_angle,
            nearest_bh_dist,
            nearest_bh_angle,
            nearest_planet_dist,
            nearest_planet_angle
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> dict:
        """Get additional info"""
        return {
            'rocket_pos': self.rocket_pos.copy(),
            'rocket_vel': self.rocket_vel.copy(),
            'fuel': self.fuel,
            'steps': self.steps
        }

    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'Astrophysics Environment - Step: {self.steps}, Fuel: {self.fuel:.1f}')

        # Draw Earth
        earth_circle = Circle(self.earth_pos, 5, color='blue', label='Earth')
        self.ax.add_patch(earth_circle)

        # Draw black holes
        for bh in self.black_holes:
            # Influence radius
            influence_circle = Circle((bh.x, bh.y), bh.influence_radius,
                                     color='purple', alpha=0.1)
            self.ax.add_patch(influence_circle)
            # Event horizon
            event_circle = Circle((bh.x, bh.y), bh.event_horizon,
                                 color='purple', alpha=0.3)
            self.ax.add_patch(event_circle)
            # Center
            center_circle = Circle((bh.x, bh.y), bh.radius, color='black')
            self.ax.add_patch(center_circle)

        # Draw suns
        for sun in self.suns:
            # Influence radius
            influence_circle = Circle((sun.x, sun.y), sun.influence_radius,
                                     color='yellow', alpha=0.1)
            self.ax.add_patch(influence_circle)
            # Center
            sun_circle = Circle((sun.x, sun.y), sun.radius, color='yellow')
            self.ax.add_patch(sun_circle)

        # Draw landable planets
        for planet in self.landable_planets:
            planet_circle = Circle((planet.x, planet.y), planet.radius, color='green')
            self.ax.add_patch(planet_circle)

        # Draw non-landable planets
        for planet in self.non_landable_planets:
            planet_circle = Circle((planet.x, planet.y), planet.radius, color='gray')
            self.ax.add_patch(planet_circle)

        # Draw asteroids
        for asteroid in self.asteroids:
            asteroid_circle = Circle((asteroid.x, asteroid.y), asteroid.radius,
                                    color='brown', alpha=0.7)
            self.ax.add_patch(asteroid_circle)

        # Draw rocket
        rocket_circle = Circle(self.rocket_pos, 3, color='red', label='Rocket')
        self.ax.add_patch(rocket_circle)

        # Draw velocity vector
        if np.linalg.norm(self.rocket_vel) > 0.1:
            self.ax.arrow(self.rocket_pos[0], self.rocket_pos[1],
                         self.rocket_vel[0] * 2, self.rocket_vel[1] * 2,
                         head_width=5, head_length=5, fc='red', ec='red', alpha=0.6)

        self.ax.legend(loc='upper right')

        if self.render_mode == 'human':
            plt.pause(0.01)
            plt.draw()
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

    def close(self):
        """Close the environment"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
