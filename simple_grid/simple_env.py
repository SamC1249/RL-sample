"""
Simple Grid World Environment for Astrophysics Simulation
100x100 grid with discrete actions and rewards
"""
import numpy as np


class SimpleGridEnv:
    """
    A simple 100x100 grid world environment.

    Features:
    - Grid size: 100x100
    - 1 black hole (reward: -100)
    - 3 planets (reward: -1 each)
    - 1 target planet (reward: 0, goal state)
    - Default cell reward: -1
    - Actions: 0=up, 1=down, 2=left, 3=right
    - Gravitational dynamics:
      * Black hole: Deterministic pull within 2-block Manhattan distance
      * Planets: 0.2 probability pull from 1-block distance when moving away
    """

    def __init__(self, grid_size=100, black_hole_radius=2, planet_radius=1, planet_pull_prob=0.2):
        self.grid_size = grid_size
        self.n_actions = 4  # up, down, left, right
        self.n_states = grid_size * grid_size

        # Gravitational parameters
        self.black_hole_radius = black_hole_radius  # Manhattan distance for black hole pull
        self.planet_radius = planet_radius  # Manhattan distance for planet pull
        self.planet_pull_prob = planet_pull_prob  # Probability of planet pull

        # Define object positions (row, col)
        self.black_hole_pos = (20, 20)  # Black hole
        self.planet_positions = [(40, 30), (60, 70), (30, 80)]  # 3 planets with -1 reward
        self.target_pos = (80, 80)  # Target planet (goal)

        # Initialize reward grid
        self.rewards = np.full((grid_size, grid_size), -1.0)

        # Set specific rewards
        self.rewards[self.black_hole_pos] = -100.0
        for pos in self.planet_positions:
            self.rewards[pos] = -1.0
        self.rewards[self.target_pos] = 0.0

        # Current state
        self.current_pos = None

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_moving_away_from(self, current_pos, next_pos, target_pos):
        """Check if moving from current_pos to next_pos is moving away from target_pos."""
        current_dist = self._manhattan_distance(current_pos, target_pos)
        next_dist = self._manhattan_distance(next_pos, target_pos)
        return next_dist > current_dist

    def reset(self, start_pos=None):
        """
        Reset the environment to starting position.

        Args:
            start_pos: Tuple (row, col) for starting position.
                      If None, starts at (0, 0)

        Returns:
            Initial state as integer (row * grid_size + col)
        """
        if start_pos is None:
            self.current_pos = (0, 0)
        else:
            self.current_pos = start_pos

        return self._pos_to_state(self.current_pos)

    def step(self, action):
        """
        Take an action in the environment with gravitational dynamics.

        Args:
            action: Integer 0-3 (0=up, 1=down, 2=left, 3=right)

        Returns:
            next_state: Integer state
            reward: Float reward
            done: Boolean indicating if episode is complete
            info: Dictionary with additional info
        """
        current_pos = self.current_pos
        row, col = current_pos

        # Execute action to get intended next position
        if action == 0:  # up
            intended_row, intended_col = max(0, row - 1), col
        elif action == 1:  # down
            intended_row, intended_col = min(self.grid_size - 1, row + 1), col
        elif action == 2:  # left
            intended_row, intended_col = row, max(0, col - 1)
        elif action == 3:  # right
            intended_row, intended_col = row, min(self.grid_size - 1, col + 1)
        else:
            intended_row, intended_col = row, col

        intended_pos = (intended_row, intended_col)

        # Apply gravitational dynamics
        final_pos, gravity_applied = self._apply_gravity(current_pos, intended_pos)

        self.current_pos = final_pos

        # Get reward
        reward = self.rewards[final_pos]

        # Check if done (reached target or black hole)
        done = final_pos == self.target_pos or final_pos == self.black_hole_pos

        next_state = self._pos_to_state(self.current_pos)

        info = {
            'position': self.current_pos,
            'reached_target': final_pos == self.target_pos,
            'hit_black_hole': final_pos == self.black_hole_pos,
            'gravity_applied': gravity_applied
        }

        return next_state, reward, done, info

    def _apply_gravity(self, current_pos, intended_pos):
        """
        Apply gravitational effects from black hole and planets.

        Args:
            current_pos: Current position (row, col)
            intended_pos: Intended next position after action (row, col)

        Returns:
            final_pos: Actual next position after gravity (row, col)
            gravity_info: String describing what gravity was applied
        """
        # Check black hole gravity (deterministic)
        # If within radius of black hole, get pulled in
        if self._manhattan_distance(current_pos, self.black_hole_pos) <= self.black_hole_radius:
            return self.black_hole_pos, 'black_hole'

        # Check planet gravity (stochastic)
        for planet_pos in self.planet_positions:
            # If currently adjacent to a planet (1 block away)
            if self._manhattan_distance(current_pos, planet_pos) == self.planet_radius:
                # If moving away from the planet
                if self._is_moving_away_from(current_pos, intended_pos, planet_pos):
                    # 20% chance of being pulled to the planet
                    if np.random.random() < self.planet_pull_prob:
                        return planet_pos, f'planet_{planet_pos}'

        # No gravity applied
        return intended_pos, 'none'

    def _pos_to_state(self, pos):
        """Convert (row, col) position to integer state."""
        row, col = pos
        return row * self.grid_size + col

    def _state_to_pos(self, state):
        """Convert integer state to (row, col) position."""
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)

    def get_transition_prob(self, state, action, next_state):
        """
        Get transition probability P(s'|s,a) with gravitational dynamics.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Probability (between 0.0 and 1.0)
        """
        current_pos = self._state_to_pos(state)
        next_pos = self._state_to_pos(next_state)
        row, col = current_pos

        # Calculate intended next position from action
        if action == 0:  # up
            intended_pos = (max(0, row - 1), col)
        elif action == 1:  # down
            intended_pos = (min(self.grid_size - 1, row + 1), col)
        elif action == 2:  # left
            intended_pos = (row, max(0, col - 1))
        elif action == 3:  # right
            intended_pos = (row, min(self.grid_size - 1, col + 1))
        else:
            return 0.0

        # Check black hole gravity (deterministic)
        if self._manhattan_distance(current_pos, self.black_hole_pos) <= self.black_hole_radius:
            # Deterministically pulled to black hole
            return 1.0 if next_pos == self.black_hole_pos else 0.0

        # Check planet gravity (stochastic)
        planet_pull_prob_total = 0.0
        pulled_to_planet = None

        for planet_pos in self.planet_positions:
            # If currently adjacent to a planet (1 block away)
            if self._manhattan_distance(current_pos, planet_pos) == self.planet_radius:
                # If moving away from the planet
                if self._is_moving_away_from(current_pos, intended_pos, planet_pos):
                    if next_pos == planet_pos:
                        planet_pull_prob_total += self.planet_pull_prob
                        pulled_to_planet = planet_pos

        # If next_state is a planet that can pull
        if pulled_to_planet is not None:
            return planet_pull_prob_total

        # If next_state is the intended position
        if next_pos == intended_pos:
            # Calculate probability of NOT being pulled by any planet
            no_pull_prob = 1.0
            for planet_pos in self.planet_positions:
                if self._manhattan_distance(current_pos, planet_pos) == self.planet_radius:
                    if self._is_moving_away_from(current_pos, intended_pos, planet_pos):
                        no_pull_prob *= (1.0 - self.planet_pull_prob)
            return no_pull_prob

        # Any other transition has 0 probability
        return 0.0

    def get_possible_next_states(self, state, action):
        """
        Get all possible next states and their probabilities for a given state-action pair.

        Args:
            state: Current state
            action: Action taken

        Returns:
            List of (next_state, probability) tuples
        """
        current_pos = self._state_to_pos(state)
        row, col = current_pos

        # Calculate intended next position from action
        if action == 0:  # up
            intended_pos = (max(0, row - 1), col)
        elif action == 1:  # down
            intended_pos = (min(self.grid_size - 1, row + 1), col)
        elif action == 2:  # left
            intended_pos = (row, max(0, col - 1))
        elif action == 3:  # right
            intended_pos = (row, min(self.grid_size - 1, col + 1))
        else:
            return []

        # Check black hole gravity (deterministic)
        if self._manhattan_distance(current_pos, self.black_hole_pos) <= self.black_hole_radius:
            black_hole_state = self._pos_to_state(self.black_hole_pos)
            return [(black_hole_state, 1.0)]

        # Check planet gravity (stochastic)
        possible_states = []
        no_pull_prob = 1.0

        for planet_pos in self.planet_positions:
            # If currently adjacent to a planet (1 block away)
            if self._manhattan_distance(current_pos, planet_pos) == self.planet_radius:
                # If moving away from the planet
                if self._is_moving_away_from(current_pos, intended_pos, planet_pos):
                    planet_state = self._pos_to_state(planet_pos)
                    possible_states.append((planet_state, self.planet_pull_prob))
                    no_pull_prob *= (1.0 - self.planet_pull_prob)

        # Add intended position with remaining probability
        intended_state = self._pos_to_state(intended_pos)
        possible_states.append((intended_state, no_pull_prob))

        return possible_states

    def get_reward(self, state):
        """Get reward for a given state."""
        row, col = self._state_to_pos(state)
        return self.rewards[row, col]

    def is_terminal(self, state):
        """Check if state is terminal (target or black hole)."""
        pos = self._state_to_pos(state)
        return pos == self.target_pos or pos == self.black_hole_pos

    def render(self, policy=None):
        """
        Simple text rendering of the environment.

        Args:
            policy: Optional policy array to visualize
        """
        grid = np.full((self.grid_size, self.grid_size), '.')

        # Mark objects
        grid[self.black_hole_pos] = 'B'
        for pos in self.planet_positions:
            grid[pos] = 'P'
        grid[self.target_pos] = 'T'

        # Mark current position
        if self.current_pos is not None:
            grid[self.current_pos] = 'A'

        # Print a subset (too large to print entire 100x100)
        print("\nGrid World (showing subset 0-30, 0-30):")
        print("B=Black Hole, P=Planet, T=Target, A=Agent, .=Empty")
        for i in range(min(30, self.grid_size)):
            print(''.join(grid[i, :min(30, self.grid_size)]))
