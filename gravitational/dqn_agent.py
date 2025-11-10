"""
Deep Q-Network (DQN) Agent for Gravitational Dynamics Environment
Handles hybrid action space by discretizing thrust magnitude.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List


class QNetwork(nn.Module):
    """Neural network for Q-value approximation"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super(QNetwork, self).__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with discretized action space.

    Action space discretization:
    - 4 directions (up, right, down, left)
    - N thrust levels (discretized from 0 to 1)
    - Total actions: 4 * N
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_directions: int = 4,
        n_thrust_levels: int = 5,
        hidden_dims: List[int] = [128, 128],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = None
    ):
        self.n_directions = n_directions
        self.n_thrust_levels = n_thrust_levels
        self.action_dim = n_directions * n_thrust_levels

        # Discretized thrust values
        self.thrust_values = np.linspace(0.0, 1.0, n_thrust_levels)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, self.action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training stats
        self.update_count = 0

    def _discretize_action(self, discrete_action: int) -> Tuple[int, float]:
        """Convert discrete action index to (direction, thrust)"""
        direction = discrete_action // self.n_thrust_levels
        thrust_idx = discrete_action % self.n_thrust_levels
        thrust = self.thrust_values[thrust_idx]
        return direction, thrust

    def _continuous_to_discrete(self, direction: int, thrust: float) -> int:
        """Convert (direction, thrust) to discrete action index"""
        thrust_idx = np.argmin(np.abs(self.thrust_values - thrust))
        return direction * self.n_thrust_levels + thrust_idx

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.

        Returns:
            action: [direction, thrust] for the environment
        """
        if training and random.random() < self.epsilon:
            # Random action
            discrete_action = random.randint(0, self.action_dim - 1)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                discrete_action = q_values.argmax().item()

        # Convert to environment action format
        direction, thrust = self._discretize_action(discrete_action)
        return np.array([direction, thrust], dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        # Convert action to discrete
        discrete_action = self._continuous_to_discrete(int(action[0]), action[1])
        self.replay_buffer.push(state, discrete_action, reward, next_state, done)

    def update(self) -> float:
        """Perform one update step using a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']


class RandomAgent:
    """Random baseline agent"""

    def __init__(self, n_directions: int = 4):
        self.n_directions = n_directions

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select random action"""
        direction = random.randint(0, self.n_directions - 1)
        thrust = random.random()
        return np.array([direction, thrust], dtype=np.float32)

    def store_transition(self, *args):
        """No-op for compatibility"""
        pass

    def update(self) -> float:
        """No-op for compatibility"""
        return 0.0

    def decay_epsilon(self):
        """No-op for compatibility"""
        pass


class HeuristicAgent:
    """
    Heuristic agent that tries to:
    1. Avoid black hole
    2. Move toward target
    3. Oppose strong gravity
    """

    def __init__(self, env):
        self.env = env
        self.target_pos = np.array([env.target_planet.x, env.target_planet.y])
        self.black_hole_pos = np.array([env.black_hole.x, env.black_hole.y])

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action based on heuristics"""
        pos = state[:2]

        # Vector to target
        to_target = self.target_pos - pos
        dist_to_target = np.linalg.norm(to_target)

        # Vector from black hole
        from_black_hole = pos - self.black_hole_pos
        dist_to_black_hole = np.linalg.norm(from_black_hole)

        # Get gravity at current position
        gravity = self.env._get_gravity_at_position(pos[0], pos[1])

        # Desired direction combines:
        # - Moving toward target
        # - Avoiding black hole if too close
        # - Opposing strong gravity

        if dist_to_black_hole < 20:  # Danger zone
            # Prioritize escaping black hole
            desired_dir = from_black_hole / (dist_to_black_hole + 1e-6)
        else:
            # Move toward target while opposing gravity
            desired_dir = to_target / (dist_to_target + 1e-6)
            if np.linalg.norm(gravity) > 0.01:
                desired_dir = 0.7 * desired_dir - 0.3 * gravity / np.linalg.norm(gravity)

        # Convert to discrete direction (up, right, down, left)
        angle = np.arctan2(desired_dir[1], desired_dir[0])

        # Map angle to direction
        # right: -π/4 to π/4
        # up: π/4 to 3π/4
        # left: 3π/4 to 5π/4 (or -3π/4 to -π/4)
        # down: -3π/4 to -π/4

        if -np.pi/4 <= angle < np.pi/4:
            direction = 1  # right
        elif np.pi/4 <= angle < 3*np.pi/4:
            direction = 0  # up
        elif angle >= 3*np.pi/4 or angle < -3*np.pi/4:
            direction = 3  # left
        else:
            direction = 2  # down

        # Thrust magnitude based on situation
        if dist_to_black_hole < 15:
            thrust = 1.0  # Maximum thrust to escape
        elif dist_to_target < 10:
            thrust = 0.3  # Gentle approach
        else:
            thrust = 0.6  # Moderate thrust

        return np.array([direction, thrust], dtype=np.float32)

    def store_transition(self, *args):
        """No-op for compatibility"""
        pass

    def update(self) -> float:
        """No-op for compatibility"""
        return 0.0

    def decay_epsilon(self):
        """No-op for compatibility"""
        pass


if __name__ == "__main__":
    # Test agent creation
    print("=== DQN Agent Test ===")
    agent = DQNAgent(
        state_dim=4,
        n_directions=4,
        n_thrust_levels=5,
        hidden_dims=[128, 128]
    )

    print(f"Device: {agent.device}")
    print(f"Total actions: {agent.action_dim}")
    print(f"Thrust levels: {agent.thrust_values}")

    # Test action selection
    state = np.array([10.0, 10.0, 0.0, 0.0])
    action = agent.select_action(state, training=True)
    print(f"\nSample action: direction={int(action[0])}, thrust={action[1]:.3f}")

    print("\n=== Random Agent Test ===")
    random_agent = RandomAgent()
    action = random_agent.select_action(state)
    print(f"Random action: direction={int(action[0])}, thrust={action[1]:.3f}")
