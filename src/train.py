



import os
import json
from collections import deque

import numpy as np
from scipy.ndimage import zoom

import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from elo_tracker import EloTracker, plot_elo_and_winrate 

def preprocess_frame(frame):
    """Preprocess frame: grayscale and resize to 84x84."""
    # Convert to grayscale (RGB -> gray)
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    # Resize from (210,160) -> (84,84)
    resized = zoom(gray, (84/210, 84/160), order=1)
    return resized.astype(np.uint8)  # keep uint8 for efficiency

class FrameStack:
    """Stack frames for temporal information (channels-first)."""
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def reset(self, frame):
        for _ in range(self.n_frames):
            self.frames.append(frame)
        return self.get_state()
    
    def step(self, frame):
        self.frames.append(frame)
        return self.get_state()
    
    def get_state(self):
        # Return shape (C,H,W)
        stacked = np.stack(self.frames, axis=0)
        return stacked

# Wrapper to expose ALE Boxing as an observation/action gym env suitable for SB3
class SB3BoxingEnv(gym.Env):
    """
    Gymnasium environment wrapper around ALE/Boxing-v5 that:
     - returns preprocessed frames
     - returns single-agent observation shaped (C,H,W) with dtype float32
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, device='cpu'):
        super().__init__()
        gym.register_envs(ale_py)

        self.raw_env = gym.make('ALE/Boxing-v5', render_mode=None)
        # We'll control player 1 only (the env's action_space is discrete)
        self.action_space = self.raw_env.action_space
        # Observation is 4 stacked grayscale frames: shape (4,84,84)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4,84,84), dtype=np.uint8)
        self.frame_stack = FrameStack(4)
    
    def reset(self, *, seed=None, options=None):
        obs, info = self.raw_env.reset(seed=seed, options=options)
        pre = preprocess_frame(obs)
        state = self.frame_stack.reset(pre)
        # SB3 likes float32 observations; we'll convert later in the vec env -> stable-baselines handles casting
        return state, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.raw_env.step(int(action))
        pre = preprocess_frame(obs)
        next_state = self.frame_stack.step(pre)
        done = bool(terminated or truncated)
        # SB3 expects (obs, reward, done, info) in Gym v0 API; gymnasium uses (obs, reward, terminated, truncated, info)
        return next_state, reward, terminated, truncated, info
    
    def render(self):
        return self.raw_env.render()
    
    def close(self):
        self.raw_env.close()



class EvalAndCheckpointCallback(BaseCallback):
    def __init__(self, eval_env, out_dir, elo_tracker=None, 
                 eval_freq_steps=50_000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.out_dir = out_dir
        self.eval_freq_steps = eval_freq_steps
        self.n_eval_episodes = n_eval_episodes
        self.elo_tracker = elo_tracker
        os.makedirs(self.out_dir, exist_ok=True)
        self.last_eval = 0
        self.best_mean_reward = -float('inf')
        self.saved_models = []
        
    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_eval) >= self.eval_freq_steps:
            self.last_eval = self.num_timesteps
            
            # Save current model
            ckpt_path = os.path.join(self.out_dir, f"model_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)
            self.saved_models.append(ckpt_path)
            
            # Do deterministic eval episodes
            rewards = []
            wins = 0
            for ep in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                ep_rew = 0.0
                while True:
                    obs_tensor = np.array(obs, dtype=np.float32)
                    action, _ = self.model.predict(obs_tensor, deterministic=True)
                    obs, r, term, trunc, info = self.eval_env.step(action)
                    ep_rew += r
                    if term or trunc:
                        break
                rewards.append(ep_rew)
                
                # Count wins (assuming positive reward means win)
                if ep_rew > 0:
                    wins += 1
            
            mean_reward = float(np.mean(rewards))
            win_rate = wins / self.n_eval_episodes
            
            # Update ELO tracker if provided
            if self.elo_tracker is not None:
                # Update ELO for each episode
                for rew in rewards:
                    win = rew > 0
                    self.elo_tracker.update(win, timestep=self.num_timesteps)
                
                current_elo = self.elo_tracker.elo
                
                # Save ELO tracker
                self.elo_tracker.save(os.path.join(self.out_dir, 'elo_tracker.json'))
            else:
                current_elo = None
            
            # Save eval stats
            stats = {
                "timesteps": self.num_timesteps, 
                "mean_reward": mean_reward, 
                "win_rate": win_rate,
                "elo": current_elo,
                "rewards": rewards
            }
            with open(os.path.join(self.out_dir, f"eval_{self.num_timesteps}.json"), "w") as f:
                json.dump(stats, f, indent=2)
            
            if self.verbose:
                elo_str = f" elo={current_elo:.1f}" if current_elo else ""
                print(f"[Eval] timesteps={self.num_timesteps} mean_reward={mean_reward:.3f} "
                      f"win_rate={win_rate:.2f}{elo_str} saved {ckpt_path}")
            
            # Track best
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = os.path.join(self.out_dir, "best_model.zip")
                self.model.save(best_path)
        
        return True
 

def train_with_sb3(total_timesteps=int(1e6),
                   outdir="runs/sb3_boxing",
                   n_envs=8,
                   eval_freq=100_000,
                   n_eval_episodes=8,
                   initial_elo=1200,
                   k=32, 
                   device="auto"):
    os.makedirs(outdir, exist_ok=True)
    
    # Create ELO tracker
    elo_tracker = EloTracker(initial_elo=initial_elo, k=k)
    
    # Make training vec env 
    venv = make_vec_env(SB3BoxingEnv, n_envs=n_envs)
    
    # Create an evaluation env (single copy)
    eval_env = SB3BoxingEnv()
    
    # Instantiate PPO
    model = PPO("CnnPolicy", venv, verbose=1, device=device, 
                tensorboard_log=os.path.join(outdir, "tensorboard"))
    
    # Callbacks: checkpoint + our eval callback with ELO tracker
    chk_cb = CheckpointCallback(save_freq=eval_freq // 2, save_path=outdir, 
                                name_prefix="ppo_boxing")
    eval_cb = EvalAndCheckpointCallback(
        eval_env=eval_env, 
        out_dir=outdir, 
        elo_tracker=elo_tracker,
        eval_freq_steps=eval_freq, 
        n_eval_episodes=n_eval_episodes
    )
    
    # Train
    print("Starting SB3 PPO training...")
    model.learn(total_timesteps=total_timesteps, callback=[chk_cb, eval_cb])
    
    # Save final model and ELO tracker
    final_path = os.path.join(outdir, "final_model.zip")
    model.save(final_path)
    elo_tracker.save(os.path.join(outdir, 'elo_tracker_final.json'))
    print(f"Saved final model to {final_path}")
    
    # Generate plots
    print("Generating ELO and win rate plots...")
    plot_elo_and_winrate(elo_tracker, save_dir=outdir)
    
    return model, eval_cb.saved_models, elo_tracker


if __name__ == "__main__":
    model, saved, elo_tracker = train_with_sb3(
        total_timesteps=500_000,
        outdir="runs/sb3_boxing",
        n_envs=1,
        eval_freq=5_000,
        n_eval_episodes=5,
        device="auto"
    )
    print("Done.")
    