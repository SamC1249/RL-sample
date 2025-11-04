"""
Q-Learning Agent for Astrophysics Environment
Uses discretization for continuous state/action spaces
"""

import numpy as np
from collections import defaultdict
import pickle
from typing import Tuple, Optional


class DiscretizedQLearningAgent:
    """
    Q-Learning agent with state and action space discretization.
    Suitable for smaller, discretized versions of the environment.
    """

    def __init__(self,
                 state_bins: Tuple[int, ...] = (20, 20, 10, 10, 10, 10, 8, 10, 8, 10, 8),
                 action_bins: Tuple[int, int] = (5, 8),
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent

        Args:
            state_bins: Number of bins for each state dimension
            action_bins: Number of bins for [thrust_magnitude, thrust_angle]
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table using defaultdict for sparse representation
        self.q_table = defaultdict(lambda: np.zeros(np.prod(action_bins)))

        # Action space
        self.thrust_values = np.linspace(0, 1, action_bins[0])
        self.angle_values = np.linspace(0, 2 * np.pi, action_bins[1])

        # State space bounds (for discretization)
        self.state_bounds = [
            (0, 1000),      # x position
            (0, 1000),      # y position
            (-50, 50),      # vx velocity
            (-50, 50),      # vy velocity
            (0, 500),       # fuel
            (0, 1000),      # nearest sun distance
            (0, 2*np.pi),   # nearest sun angle
            (0, 1000),      # nearest black hole distance
            (0, 2*np.pi),   # nearest black hole angle
            (0, 1000),      # nearest planet distance
            (0, 2*np.pi),   # nearest planet angle
        ]

        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_values': []
        }

    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretize continuous state into bins"""
        discretized = []
        for i, (low, high) in enumerate(self.state_bounds):
            # Clip state to bounds
            value = np.clip(state[i], low, high)
            # Normalize to [0, 1]
            normalized = (value - low) / (high - low + 1e-10)
            # Discretize
            bin_idx = int(normalized * (self.state_bins[i] - 1))
            discretized.append(bin_idx)
        return tuple(discretized)

    def discretize_action(self, action_idx: int) -> np.ndarray:
        """Convert discrete action index to continuous action"""
        thrust_idx = action_idx // self.action_bins[1]
        angle_idx = action_idx % self.action_bins[1]

        thrust = self.thrust_values[thrust_idx]
        angle = self.angle_values[angle_idx]

        return np.array([thrust, angle], dtype=np.float32)

    def get_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            training: Whether in training mode (exploration) or evaluation mode

        Returns:
            Continuous action [thrust_magnitude, thrust_angle]
        """
        # Discretize state
        discrete_state = self.discretize_state(state)

        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            # Random action
            action_idx = np.random.randint(0, np.prod(self.action_bins))
        else:
            # Greedy action
            q_values = self.q_table[discrete_state]
            action_idx = np.argmax(q_values)

        # Convert to continuous action
        return self.discretize_action(action_idx)

    def update(self, state: np.ndarray, action: np.ndarray,
               reward: float, next_state: np.ndarray, done: bool):
        """
        Update Q-values using Q-learning update rule

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Discretize states
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)

        # Find action index
        action_idx = self._find_closest_action_idx(action)

        # Q-learning update
        current_q = self.q_table[discrete_state][action_idx]

        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[discrete_next_state])
            target_q = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[discrete_state][action_idx] += self.lr * (target_q - current_q)

    def _find_closest_action_idx(self, action: np.ndarray) -> int:
        """Find the closest discrete action index for a continuous action"""
        thrust, angle = action[0], action[1]

        # Find closest thrust bin
        thrust_idx = np.argmin(np.abs(self.thrust_values - thrust))

        # Find closest angle bin
        angle_idx = np.argmin(np.abs(self.angle_values - angle))

        return thrust_idx * self.action_bins[1] + angle_idx

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save Q-table and agent parameters"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'training_stats': self.training_stats,
                'state_bins': self.state_bins,
                'action_bins': self.action_bins
            }, f)

    def load(self, filepath: str):
        """Load Q-table and agent parameters"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(np.prod(self.action_bins)),
                                      data['q_table'])
            self.epsilon = data['epsilon']
            self.training_stats = data['training_stats']


def train_q_learning(env, agent, num_episodes: int = 1000,
                    max_steps_per_episode: int = 2000,
                    save_path: Optional[str] = None) -> DiscretizedQLearningAgent:
    """
    Train Q-Learning agent

    Args:
        env: Gym environment
        agent: Q-Learning agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_path: Path to save trained agent

    Returns:
        Trained agent
    """
    print(f"Training Q-Learning agent for {num_episodes} episodes...")

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Select action
            action = agent.get_action(state, training=True)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update Q-values
            agent.update(state, action, reward, next_state, done)

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

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
            avg_length = np.mean(agent.training_stats['episode_lengths'][-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

        # Save checkpoint
        if save_path and (episode + 1) % 100 == 0:
            agent.save(f"{save_path}_episode_{episode + 1}.pkl")

    # Save final model
    if save_path:
        agent.save(f"{save_path}_final.pkl")
        print(f"Saved trained agent to {save_path}_final.pkl")

    return agent


if __name__ == "__main__":
    # Example usage
    from astrophysics_env import AstrophysicsEnv

    # Create environment
    env = AstrophysicsEnv(grid_size=1000, max_fuel=500.0, max_steps=500, seed=42)

    # Create agent
    agent = DiscretizedQLearningAgent(
        state_bins=(20, 20, 10, 10, 10, 10, 8, 10, 8, 10, 8),
        action_bins=(5, 8),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Train agent
    trained_agent = train_q_learning(
        env=env,
        agent=agent,
        num_episodes=500,
        max_steps_per_episode=500,
        save_path="models/q_learning_agent"
    )

    print("\nTraining completed!")
