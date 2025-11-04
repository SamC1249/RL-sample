"""
Proximal Policy Optimization (PPO) Agent for Astrophysics Environment
Handles continuous action space directly using policy gradient methods
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Optional, List


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    Actor outputs mean and std for continuous actions
    Critic outputs state value
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        """
        Initialize Actor-Critic network

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: Sizes of hidden layers
        """
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim

        # Shared layers
        shared_layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes[:-1]:
            shared_layers.append(nn.Linear(prev_size, hidden_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.LayerNorm(hidden_size))
            prev_size = hidden_size

        self.shared = nn.Sequential(*shared_layers)

        # Actor head (policy)
        self.actor_hidden = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[-1])
        )
        self.actor_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic_hidden = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[-1])
        )
        self.critic = nn.Linear(hidden_sizes[-1], 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """Forward pass (not used directly, use act() and evaluate() instead)"""
        shared_features = self.shared(state)
        return shared_features

    def act(self, state):
        """
        Sample action from policy

        Args:
            state: Current state

        Returns:
            action, log_prob, value
        """
        shared_features = self.shared(state)

        # Actor
        actor_features = self.actor_hidden(shared_features)
        action_mean = self.actor_mean(actor_features)
        action_std = torch.exp(self.actor_log_std)

        # Sample action
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Critic
        critic_features = self.critic_hidden(shared_features)
        value = self.critic(critic_features)

        return action, log_prob, value

    def evaluate(self, state, action):
        """
        Evaluate actions

        Args:
            state: States
            action: Actions

        Returns:
            log_probs, values, entropy
        """
        shared_features = self.shared(state)

        # Actor
        actor_features = self.actor_hidden(shared_features)
        action_mean = self.actor_mean(actor_features)
        action_std = torch.exp(self.actor_log_std)

        # Evaluate action
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        # Critic
        critic_features = self.critic_hidden(shared_features)
        values = self.critic(critic_features)

        return log_probs, values, entropy


class RolloutBuffer:
    """Buffer for storing trajectories"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, log_prob, value, done):
        """Add experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        """Clear buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def get(self):
        """Get all data as tensors"""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones)
        )

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    Proximal Policy Optimization agent
    Handles continuous action space directly
    """

    def __init__(self,
                 state_dim: int = 11,
                 action_dim: int = 2,
                 learning_rate: float = 3e-4,
                 discount_factor: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 num_epochs: int = 10,
                 batch_size: int = 64,
                 device: str = 'auto'):
        """
        Initialize PPO agent

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (2 for thrust and angle)
            learning_rate: Learning rate
            discount_factor: Discount factor (gamma)
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of epochs per update
            batch_size: Batch size for training
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Actor-Critic network
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': []
        }

    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Sample action from policy

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.actor_critic.act(state_tensor)

            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().item()
            value = value.cpu().item()

        # Post-process action
        # Thrust magnitude: clip to [0, 1]
        action[0] = np.clip(action[0], 0, 1)
        # Angle: wrap to [0, 2π]
        action[1] = action[1] % (2 * np.pi)

        return action, log_prob, value

    def store_experience(self, state, action, reward, log_prob, value, done):
        """Store experience in buffer"""
        self.buffer.add(state, action, reward, log_prob, value, done)

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: Rewards
            values: Value estimates
            dones: Done flags
            next_value: Value of next state

        Returns:
            advantages, returns
        """
        advantages = []
        gae = 0

        values = values.tolist() + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)

        return advantages, returns

    def update(self, next_state):
        """
        Update policy using PPO

        Args:
            next_state: Next state after rollout

        Returns:
            Dictionary of losses
        """
        # Get rollout data
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get()

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # Compute next value
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, _, next_value = self.actor_critic.act(next_state_tensor)
            next_value = next_value.cpu().item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.num_epochs):
            # Generate random indices
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, state_values, entropy = self.actor_critic.evaluate(
                    batch_states, batch_actions
                )

                # Compute ratios
                ratios = torch.exp(log_probs - batch_old_log_probs)

                # Compute surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Record stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # Clear buffer
        self.buffer.clear()

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']


def train_ppo(env, agent, num_episodes: int = 1000,
              max_steps_per_episode: int = 2000,
              update_interval: int = 2048,
              save_path: Optional[str] = None) -> PPOAgent:
    """
    Train PPO agent

    Args:
        env: Gym environment
        agent: PPO agent
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        update_interval: Number of steps before policy update
        save_path: Path to save trained agent

    Returns:
        Trained agent
    """
    print(f"Training PPO agent on {agent.device} for {num_episodes} episodes...")

    total_steps = 0
    episode = 0

    while episode < num_episodes:
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Select action
            action, log_prob, value = agent.get_action(state, training=True)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(state, action, reward, log_prob, value, done)

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Update policy
            if total_steps % update_interval == 0:
                losses = agent.update(state)
                agent.training_stats['policy_losses'].append(losses['policy_loss'])
                agent.training_stats['value_losses'].append(losses['value_loss'])
                agent.training_stats['entropies'].append(losses['entropy'])

            if done:
                break

        # Record statistics
        agent.training_stats['episode_rewards'].append(episode_reward)
        agent.training_stats['episode_lengths'].append(episode_length)

        episode += 1

        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(agent.training_stats['episode_rewards'][-10:])
            avg_length = np.mean(agent.training_stats['episode_lengths'][-10:])
            if agent.training_stats['policy_losses']:
                avg_policy_loss = np.mean(agent.training_stats['policy_losses'][-10:])
                avg_value_loss = np.mean(agent.training_stats['value_losses'][-10:])
                avg_entropy = np.mean(agent.training_stats['entropies'][-10:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Policy Loss: {avg_policy_loss:.4f} | "
                      f"Value Loss: {avg_value_loss:.4f} | "
                      f"Entropy: {avg_entropy:.4f}")
            else:
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f}")

        # Save checkpoint
        if save_path and episode % 100 == 0:
            agent.save(f"{save_path}_episode_{episode}.pt")

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
    agent = PPOAgent(
        state_dim=11,
        action_dim=2,
        learning_rate=3e-4,
        discount_factor=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        num_epochs=10,
        batch_size=64
    )

    # Train agent
    trained_agent = train_ppo(
        env=env,
        agent=agent,
        num_episodes=500,
        max_steps_per_episode=500,
        update_interval=2048,
        save_path="models/ppo_agent"
    )

    print("\nTraining completed!")
