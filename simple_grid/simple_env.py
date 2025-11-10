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
    - No gravitational dynamics
    """

    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.n_actions = 4  # up, down, left, right
        self.n_states = grid_size * grid_size

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
        Take an action in the environment.

        Args:
            action: Integer 0-3 (0=up, 1=down, 2=left, 3=right)

        Returns:
            next_state: Integer state
            reward: Float reward
            done: Boolean indicating if episode is complete
            info: Dictionary with additional info
        """
        row, col = self.current_pos

        # Execute action
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.grid_size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.grid_size - 1, col + 1)

        self.current_pos = (row, col)

        # Get reward
        reward = self.rewards[row, col]

        # Check if done (reached target or black hole)
        done = (row, col) == self.target_pos or (row, col) == self.black_hole_pos

        next_state = self._pos_to_state(self.current_pos)

        info = {
            'position': self.current_pos,
            'reached_target': (row, col) == self.target_pos,
            'hit_black_hole': (row, col) == self.black_hole_pos
        }

        return next_state, reward, done, info

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
        Get transition probability P(s'|s,a).
        Since this is deterministic, returns 1.0 for valid transitions, 0.0 otherwise.

        Args:
            state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Probability (0.0 or 1.0)
        """
        row, col = self._state_to_pos(state)

        # Calculate expected next position
        if action == 0:  # up
            expected_pos = (max(0, row - 1), col)
        elif action == 1:  # down
            expected_pos = (min(self.grid_size - 1, row + 1), col)
        elif action == 2:  # left
            expected_pos = (row, max(0, col - 1))
        elif action == 3:  # right
            expected_pos = (row, min(self.grid_size - 1, col + 1))
        else:
            return 0.0

        expected_state = self._pos_to_state(expected_pos)

        return 1.0 if expected_state == next_state else 0.0

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
