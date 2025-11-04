"""
Deep Q-Network (DQN) Agent for Astrophysics Environment
Uses neural networks to handle continuous state space with discretized actions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Tuple, Optional


# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_dim: int, num_actions: int, hidden_sizes: Tuple[int, ...] = (256, 256, 128)):
        """
        Initialize DQN

        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions
            hidden_sizes: Sizes of hidden layers
        """
        super(DQNetwork, self).__init__()

        layers = []
        prev_size = state_dim

        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass"""
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([e.done for e in experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network
    """

    def __init__(self,
                 state_dim: int = 11,
                 action_bins: Tuple[int, int] = (7, 12),
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 10,
                 device: str = 'auto'):
        """
        Initialize DQN agent

        Args:
            state_dim: Dimension of state space
            action_bins: Number of bins for [thrust_magnitude, thrust_angle]
            learning_rate: Learning rate
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.state_dim = state_dim
        self.action_bins = action_bins
        self.num_actions = action_bins[0] * action_bins[1]
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Networks
        self.q_network = DQNetwork(state_dim, self.num_actions).to(self.device)
        self.target_network = DQNetwork(state_dim, self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Action space
        self.thrust_values = np.linspace(0, 1, action_bins[0])
        self.angle_values = np.linspace(0, 2 * np.pi, action_bins[1])

        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': [],
            'losses': []
        }
        self.update_counter = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, int]:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Tuple of (continuous action, action index)
        """
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            # Random action
            action_idx = np.random.randint(0, self.num_actions)
        else:
            # Greedy action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax().item()

        # Convert to continuous action
        continuous_action = self._idx_to_action(action_idx)
        return continuous_action, action_idx

    def _idx_to_action(self, action_idx: int) -> np.ndarray:
        """Convert action index to continuous action"""
        thrust_idx = action_idx // self.action_bins[1]
        angle_idx = action_idx % self.action_bins[1]

        thrust = self.thrust_values[thrust_idx]
        angle = self.angle_values[angle_idx]

        return np.array([thrust, angle], dtype=np.float32)

    def store_experience(self, state, action_idx, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action_idx, reward, next_state, done)

    def update(self):
        """
        Update Q-network using experience replay

        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_stats = checkpoint['training_stats']


def train_dqn(env, agent, num_episodes: int = 1000,
              max_steps_per_episode: int = 2000,
              update_freq: int = 4,
              save_path: Optional[str] = None) -> DQNAgent:
    """
    Train DQN agent

    Args:
        env: Gym environment
        agent: DQN agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        update_freq: Frequency of network updates
        save_path: Path to save trained agent

    Returns:
        Trained agent
    """
    print(f"Training DQN agent on {agent.device} for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []

        for step in range(max_steps_per_episode):
            # Select action
            action, action_idx = agent.get_action(state, training=True)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(state, action_idx, reward, next_state, done)

            # Update network
            if step % update_freq == 0:
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Record statistics
        agent.training_stats['episode_rewards'].append(episode_reward)
        agent.training_stats['episode_lengths'].append(episode_length)
        agent.training_stats['epsilon_values'].append(agent.epsilon)
        if episode_losses:
            agent.training_stats['losses'].append(np.mean(episode_losses))

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
            avg_length = np.mean(agent.training_stats['episode_lengths'][-10:])
            avg_loss = np.mean(agent.training_stats['losses'][-10:]) if agent.training_stats['losses'] else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer)}")

        # Save checkpoint
        if save_path and (episode + 1) % 100 == 0:
            agent.save(f"{save_path}_episode_{episode + 1}.pt")

    # Save final model
    if save_path:
        agent.save(f"{save_path}_final.pt")
        print(f"Saved trained agent to {save_path}_final.pt")

    return agent


if __name__ == "__main__":
    # Example usage
    from astrophysics_env import AstrophysicsEnv

    # Create environment
    env = AstrophysicsEnv(grid_size=1000, max_fuel=500.0, max_steps=500, seed=42)

    # Create agent
    agent = DQNAgent(
        state_dim=11,
        action_bins=(7, 12),
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_capacity=50000,
        batch_size=64,
        target_update_freq=10
    )

    # Train agent
    trained_agent = train_dqn(
        env=env,
        agent=agent,
        num_episodes=500,
        max_steps_per_episode=500,
        update_freq=4,
        save_path="models/dqn_agent"
    )

    print("\nTraining completed!")
