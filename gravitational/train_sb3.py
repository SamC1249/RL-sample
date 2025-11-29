"""
Fast DQN Training using Stable-Baselines3
Professional implementation with optimized training loop and logging.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gravitational_env import GravitationalDynamicsEnv
import os
from datetime import datetime
import json


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback for logging training progress with detailed metrics.
    """
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.black_hole_count = 0
        self.episode_count = 0
        
        # For current episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """Called at each step"""
        # Track episode statistics
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode ended
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Check terminal reason from info
            info = self.locals['infos'][0]
            if 'terminal_reason' in info:
                if info['terminal_reason'] == 'success':
                    self.success_count += 1
                elif info['terminal_reason'] == 'black_hole':
                    self.black_hole_count += 1
            
            # Log to tensorboard
            self.logger.record('rollout/ep_reward', self.current_episode_reward)
            self.logger.record('rollout/ep_length', self.current_episode_length)
            self.logger.record('rollout/success_rate', self.success_count / max(1, self.episode_count))
            self.logger.record('rollout/black_hole_rate', self.black_hole_count / max(1, self.episode_count))
            
            # Print progress
            if self.episode_count % 50 == 0 and self.verbose > 0:
                recent_window = min(100, len(self.episode_rewards))
                recent_rewards = self.episode_rewards[-recent_window:]
                recent_success = sum(1 for i in range(-recent_window, 0) 
                                    if i >= -len(self.episode_rewards)) / recent_window
                
                print(f"\n{'='*80}")
                print(f"Episode {self.episode_count} | Steps: {self.num_timesteps}")
                print(f"{'='*80}")
                print(f"  Avg Reward (recent):       {np.mean(recent_rewards):>8.2f}")
                print(f"  Success Rate (overall):    {self.success_count / self.episode_count:>8.2%}")
                print(f"  Success Rate (recent):     {recent_success:>8.2%}")
                print(f"  Black Hole Rate:           {self.black_hole_count / self.episode_count:>8.2%}")
                print(f"  Current Episode Reward:    {self.current_episode_reward:>8.2f}")
                print(f"{'='*80}\n")
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


def make_env():
    """Create and wrap the environment"""
    from wrappers import make_wrapped_env
    
    # Create environment with discrete action space for DQN
    env = make_wrapped_env(
        normalize_obs=True,      # Normalize observations to [-1, 1]
        reward_shaping=False,    # Optional: add distance-based rewards
        n_thrust_levels=5        # Discretize thrust into 5 levels
    )
    env = Monitor(env)  # Wrap with Monitor for automatic logging
    return env


def train_dqn_sb3(
    total_timesteps: int = 500000,
    learning_rate: float = 1e-4,
    buffer_size: int = 100000,
    learning_starts: int = 10000,
    batch_size: int = 128,
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.01,
    train_freq: int = 4,
    gradient_steps: int = 1,
    save_dir: str = "checkpoints/sb3_dqn",
    verbose: int = 1
):
    """
    Train DQN agent using Stable-Baselines3.
    
    This is a professional, highly optimized implementation with:
    - Vectorized operations
    - Efficient replay buffer
    - Proper exploration scheduling
    - Tensorboard logging
    - Automatic checkpointing
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print("="*80)
    print("Training DQN with Stable-Baselines3")
    print("="*80)
    print(f"Total timesteps:        {total_timesteps:,}")
    print(f"Learning rate:          {learning_rate}")
    print(f"Buffer size:            {buffer_size:,}")
    print(f"Batch size:             {batch_size}")
    print(f"Target update interval: {target_update_interval}")
    print(f"Train frequency:        {train_freq}")
    print(f"Exploration fraction:   {exploration_fraction}")
    print(f"Save directory:         {save_dir}")
    print("="*80)
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Create DQN model with optimized hyperparameters
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=1.0,  # Hard update (set to 0.005 for soft updates)
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=dict(net_arch=[128, 128]),  # Network architecture
        tensorboard_log=log_dir,
        verbose=verbose,
        device="auto"  # Automatically use GPU if available
    )
    
    print(f"\nUsing device: {model.device}")
    print(f"Policy architecture: {model.policy}")
    
    # Create callbacks
    progress_callback = TrainingProgressCallback(eval_freq=1000, verbose=verbose)
    
    # Checkpoint callback - save every 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="dqn_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Evaluation callback - evaluate every 10k steps
    eval_env = DummyVecEnv([make_env])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print("\nStarting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_callback, checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\n{'='*80}")
    print(f"Training complete! Final model saved to: {final_path}")
    print(f"{'='*80}")
    
    # Save training statistics (convert numpy types to native Python types for JSON)
    stats = {
        'total_timesteps': int(total_timesteps),
        'total_episodes': int(progress_callback.episode_count),
        'success_count': int(progress_callback.success_count),
        'black_hole_count': int(progress_callback.black_hole_count),
        'success_rate': float(progress_callback.success_count / max(1, progress_callback.episode_count)),
        'black_hole_rate': float(progress_callback.black_hole_count / max(1, progress_callback.episode_count)),
        'avg_episode_reward': float(np.mean(progress_callback.episode_rewards)) if progress_callback.episode_rewards else 0.0,
        'avg_episode_length': float(np.mean(progress_callback.episode_lengths)) if progress_callback.episode_lengths else 0.0
    }
    
    stats_path = os.path.join(save_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nFinal Statistics:")
    print(f"  Total Episodes:     {stats['total_episodes']}")
    print(f"  Success Rate:       {stats['success_rate']:.2%}")
    print(f"  Black Hole Rate:    {stats['black_hole_rate']:.2%}")
    print(f"  Avg Episode Reward: {stats['avg_episode_reward']:.2f}")
    print(f"  Avg Episode Length: {stats['avg_episode_length']:.2f}")
    
    print(f"\nTo view training progress in TensorBoard, run:")
    print(f"  tensorboard --logdir {log_dir}")
    
    return model, stats


def evaluate_model(model_path: str, n_episodes: int = 100, render: bool = False):
    """Evaluate a trained model"""
    print(f"\nEvaluating model: {model_path}")
    
    # Load model
    model = DQN.load(model_path)
    
    # Create environment
    env = make_env()
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    black_hole_count = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                env.render()
            
            if terminated:
                if info.get('terminal_reason') == 'success':
                    success_count += 1
                elif info.get('terminal_reason') == 'black_hole':
                    black_hole_count += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    # Print results
    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    print(f"Episodes:           {n_episodes}")
    print(f"Avg Reward:         {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Avg Length:         {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Success Rate:       {success_count / n_episodes:.2%}")
    print(f"Black Hole Rate:    {black_hole_count / n_episodes:.2%}")
    print(f"{'='*80}")
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'success_rate': success_count / n_episodes,
        'black_hole_rate': black_hole_count / n_episodes
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN with Stable-Baselines3")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--save-dir", type=str, default="checkpoints/sb3_dqn", help="Save directory")
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    
    args = parser.parse_args()
    
    if args.eval:
        if not args.model_path:
            print("Error: --model-path required for evaluation")
            exit(1)
        evaluate_model(args.model_path, n_episodes=100)
    else:
        model, stats = train_dqn_sb3(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            verbose=1
        )

