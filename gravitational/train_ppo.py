"""
Fast PPO Training using Stable-Baselines3
PPO handles continuous/hybrid action spaces natively - better for this environment!
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gravitational_env import GravitationalDynamicsEnv
import os
from datetime import datetime
import json


class TrainingProgressCallback(BaseCallback):
    """Custom callback for logging training progress"""
    def __init__(self, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.black_hole_count = 0
        self.episode_count = 0
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """Called at each step"""
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            info = self.locals['infos'][0]
            if 'terminal_reason' in info:
                if info['terminal_reason'] == 'success':
                    self.success_count += 1
                elif info['terminal_reason'] == 'black_hole':
                    self.black_hole_count += 1
            
            self.logger.record('rollout/ep_reward', self.current_episode_reward)
            self.logger.record('rollout/ep_length', self.current_episode_length)
            self.logger.record('rollout/success_rate', self.success_count / max(1, self.episode_count))
            self.logger.record('rollout/black_hole_rate', self.black_hole_count / max(1, self.episode_count))
            
            if self.episode_count % 50 == 0 and self.verbose > 0:
                recent_window = min(100, len(self.episode_rewards))
                recent_rewards = self.episode_rewards[-recent_window:]
                
                print(f"\n{'='*80}")
                print(f"Episode {self.episode_count} | Steps: {self.num_timesteps}")
                print(f"{'='*80}")
                print(f"  Avg Reward (recent):       {np.mean(recent_rewards):>8.2f}")
                print(f"  Success Rate (overall):    {self.success_count / self.episode_count:>8.2%}")
                print(f"  Black Hole Rate:           {self.black_hole_count / self.episode_count:>8.2%}")
                print(f"  Current Episode Reward:    {self.current_episode_reward:>8.2f}")
                print(f"{'='*80}\n")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


def make_env():
    """Create and wrap the environment - PPO handles Box action space natively!"""
    env = GravitationalDynamicsEnv(grid_size=100, G=1e-3, k=1.0)
    env = Monitor(env)
    return env


def train_ppo_sb3(
    total_timesteps: int = 500000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    save_dir: str = "checkpoints/sb3_ppo",
    verbose: int = 1
):
    """
    Train PPO agent using Stable-Baselines3.
    
    PPO is often BETTER than DQN for continuous/hybrid action spaces!
    - Handles Box action space natively (no discretization needed)
    - More stable training
    - Often faster convergence
    """
    
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print("="*80)
    print("Training PPO with Stable-Baselines3")
    print("="*80)
    print(f"Total timesteps:    {total_timesteps:,}")
    print(f"Learning rate:      {learning_rate}")
    print(f"Steps per rollout:  {n_steps}")
    print(f"Batch size:         {batch_size}")
    print(f"Epochs per update:  {n_epochs}")
    print(f"Save directory:     {save_dir}")
    print("="*80)
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])  # Policy and value networks
        ),
        tensorboard_log=log_dir,
        verbose=verbose,
        device="auto"
    )
    
    print(f"\nUsing device: {model.device}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Create callbacks
    progress_callback = TrainingProgressCallback(eval_freq=1000, verbose=verbose)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,  # PPO doesn't use replay buffer
        save_vecnormalize=True
    )
    
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
    
    # Train
    print("\nStarting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_callback, checkpoint_callback, eval_callback],
            log_interval=10,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⚠️  Training interrupted by user!")
        print("="*80)
        # Save the model even if interrupted
        interrupted_path = os.path.join(save_dir, "interrupted_model")
        model.save(interrupted_path)
        print(f"✓ Model saved to: {interrupted_path}.zip")
        print("="*80 + "\n")
        
        # Create partial stats
        stats = {
            'total_timesteps': int(model.num_timesteps),
            'total_episodes': int(progress_callback.episode_count),
            'success_count': int(progress_callback.success_count),
            'black_hole_count': int(progress_callback.black_hole_count),
            'success_rate': float(progress_callback.success_count / max(1, progress_callback.episode_count)),
            'black_hole_rate': float(progress_callback.black_hole_count / max(1, progress_callback.episode_count)),
            'avg_episode_reward': float(np.mean(progress_callback.episode_rewards)) if progress_callback.episode_rewards else 0.0,
            'avg_episode_length': float(np.mean(progress_callback.episode_lengths)) if progress_callback.episode_lengths else 0.0
        }
        
        return model, stats
    
    # Save final model
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\n{'='*80}")
    print(f"Training complete! Final model saved to: {final_path}.zip")
    print(f"{'='*80}")
    
    # Save statistics (convert numpy types to native Python types for JSON)
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
    
    print(f"\nTo view training progress in TensorBoard, run:")
    print(f"  tensorboard --logdir {log_dir}")
    
    return model, stats


def train_ppo_sb3_with_interrupt_handling(*args, **kwargs):
    """
    Wrapper for train_ppo_sb3 that handles KeyboardInterrupt gracefully.
    Returns the model even if training is interrupted.
    """
    try:
        return train_ppo_sb3(*args, **kwargs)
    except KeyboardInterrupt:
        # Re-raise to be handled by caller
        raise


def evaluate_model(model_path: str, n_episodes: int = 100, render: bool = False):
    """Evaluate a trained PPO model"""
    print(f"\nEvaluating model: {model_path}")
    
    model = PPO.load(model_path)
    env = make_env()
    
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
    
    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    print(f"Episodes:           {n_episodes}")
    print(f"Avg Reward:         {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
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
    
    parser = argparse.ArgumentParser(description="Train PPO with Stable-Baselines3")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="checkpoints/sb3_ppo", help="Save directory")
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--model-path", type=str, help="Path to model for evaluation")
    
    args = parser.parse_args()
    
    if args.eval:
        if not args.model_path:
            print("Error: --model-path required for evaluation")
            exit(1)
        evaluate_model(args.model_path, n_episodes=100)
    else:
        model, stats = train_ppo_sb3(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            verbose=1
        )

