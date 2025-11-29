"""
Quick training script - just run this!
Uses PPO (better for hybrid action spaces) with Stable-Baselines3.
"""

import os

# Choose algorithm: "ppo" (recommended) or "dqn"
ALGORITHM = "ppo"

if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"QUICK TRAIN - {ALGORITHM.upper()} with Stable-Baselines3")
    print("="*80 + "\n")
    
    if ALGORITHM == "ppo":
        from train_ppo import train_ppo_sb3, evaluate_model
        
        print("Using PPO (Proximal Policy Optimization)")
        print("  ✅ Handles continuous/hybrid actions natively")
        print("  ✅ More stable training")
        print("  ✅ Often faster convergence\n")
        
        model, stats = train_ppo_sb3(
            total_timesteps=500000,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            save_dir="checkpoints/sb3_ppo_quick",
            verbose=1
        )
        
        save_dir = "checkpoints/sb3_ppo_quick"
        
    else:  # dqn
        from train_sb3 import train_dqn_sb3, evaluate_model
        
        print("Using DQN (Deep Q-Network)")
        print("  ⚠️  Requires action space discretization")
        print("  ⚠️  May be slower for this problem\n")
        
        model, stats = train_dqn_sb3(
            total_timesteps=500000,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=128,
            save_dir="checkpoints/sb3_dqn_quick",
            verbose=1
        )
        
        save_dir = "checkpoints/sb3_dqn_quick"
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nSuccess Rate: {stats['success_rate']:.2%}")
    print(f"Black Hole Rate: {stats['black_hole_rate']:.2%}")
    print(f"Avg Reward: {stats['avg_episode_reward']:.2f}")
    
    # Evaluate the trained model
    print("\n" + "="*80)
    print("Evaluating trained model on 50 episodes...")
    print("="*80 + "\n")
    
    model_path = os.path.join(save_dir, "final_model")
    eval_stats = evaluate_model(model_path, n_episodes=50, render=False)
    
    print("\n" + "="*80)
    print("DONE! To view training curves, run:")
    print(f"  tensorboard --logdir {save_dir}/logs")
    print("="*80 + "\n")

