# train_sb3_boxing.py
import os
import time
import json
import random
from collections import deque, defaultdict

import numpy as np
import torch
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Keep your preprocessing and frame stacking
from scipy.ndimage import zoom

def preprocess_frame(frame):
    """Preprocess frame: grayscale and resize to 84x84 (uint8)."""
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

# Minimal ELO tracker placeholder to integrate with your pipeline
# If you have your own elo_tracker module, import that instead.
class ELOTrackerSimple:
    def __init__(self, k_factor=32, initial_elo=1500):
        self.k = k_factor
        self.current_elo = initial_elo
        self.history = []
        self.win_history = []
    
    @staticmethod
    def expected(a, b):
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))
    
    def update_from_result(self, our_reward, opponent_reward):
        # Heuristic: treat reward>0 as win, =0 draw, <0 loss (depends on env reward scale)
        if our_reward > opponent_reward:
            score = 1.0
        elif our_reward == opponent_reward:
            score = 0.5
        else:
            score = 0.0
        exp = self.expected(self.current_elo, 1500.0)  # we compare vs baseline-opponent rating 1500
        self.current_elo = self.current_elo + self.k * (score - exp)
        self.history.append(self.current_elo)
        self.win_history.append(score)
        return self.current_elo, score
    
    def get_stats(self, window=10):
        wins = self.win_history[-window:]
        avg_reward = None
        return {"current_elo": self.current_elo, "win_rate": (np.mean(wins) if len(wins)>0 else 0.0)}

# Callback to do periodic evaluation and checkpointing
class EvalAndCheckpointCallback(BaseCallback):
    def __init__(self, eval_env, out_dir, eval_freq_steps=50_000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.out_dir = out_dir
        self.eval_freq_steps = eval_freq_steps
        self.n_eval_episodes = n_eval_episodes
        os.makedirs(self.out_dir, exist_ok=True)
        self.last_eval = 0
        self.best_mean_reward = -float('inf')
        # store saved checkpoints for offline analysis
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
            for ep in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                done = False
                ep_rew = 0.0
                # we must pass obs through the same preprocessing (the env already provides stacked uint8)
                while True:
                    # SB3 expects either flattened or vectorized obs -> convert to float32 and add batch dim
                    obs_tensor = np.array(obs, dtype=np.float32)
                    action, _ = self.model.predict(obs_tensor, deterministic=True)
                    obs, r, term, trunc, info = self.eval_env.step(action)
                    ep_rew += r
                    if term or trunc:
                        break
                rewards.append(ep_rew)
            mean_reward = float(np.mean(rewards))
            # Save eval stats
            stats = {"timesteps": self.num_timesteps, "mean_reward": mean_reward, "rewards": rewards}
            with open(os.path.join(self.out_dir, f"eval_{self.num_timesteps}.json"), "w") as f:
                json.dump(stats, f, indent=2)
            if self.verbose:
                print(f"[Eval] timesteps={self.num_timesteps} mean_reward={mean_reward:.3f} saved {ckpt_path}")
            # track best
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Keep best model copy name
                best_path = os.path.join(self.out_dir, "best_model.zip")
                self.model.save(best_path)
        return True

def make_vec_env_for_sb3(n_envs=4):
    """Create vectorized envs for SB3 training using our SB3BoxingEnv."""
    def _make():
        return SB3BoxingEnv()
    # Create a DummyVecEnv with multiple copies for parallelization
    venv = DummyVecEnv([_make for _ in range(n_envs)])
    # We rely on SB3 VecFrameStack to handle stacking if desired, but our wrapper already stacks 4 frames.
    # So do not apply VecFrameStack again. Convert observations to float32 + normalize optionally.
    # Optionally wrap with VecNormalize if desired (unstable for Atari raw rewards).
    return venv

def train_with_sb3(total_timesteps=int(1e6),
                   outdir="runs/sb3_boxing",
                   n_envs=8,
                   eval_freq=100_000,
                   n_eval_episodes=8,
                   elo_tracker=None,
                   device="auto"):
    os.makedirs(outdir, exist_ok=True)
    # Make training vec env
    venv = make_vec_env_for_sb3(n_envs=n_envs)
    # Note: our env returns uint8 arrays of shape (4,84,84). SB3 will convert dtypes automatically on learning/predict.
    # Create an evaluation env (single copy)
    eval_env = SB3BoxingEnv()
    # Instantiate PPO
    model = PPO("CnnPolicy", venv, verbose=1, device=device, tensorboard_log=os.path.join(outdir, "tensorboard"))
    # Callbacks: checkpoint + our eval callback
    chk_cb = CheckpointCallback(save_freq=eval_freq // 2, save_path=outdir, name_prefix="ppo_boxing")
    eval_cb = EvalAndCheckpointCallback(eval_env=eval_env, out_dir=outdir, eval_freq_steps=eval_freq, n_eval_episodes=n_eval_episodes)
    # Train
    print("Starting SB3 PPO training...")
    model.learn(total_timesteps=total_timesteps, callback=[chk_cb, eval_cb])
    # Save final model
    final_path = os.path.join(outdir, "final_model.zip")
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    return model, eval_cb.saved_models

if __name__ == "__main__":
    # small CLI-style defaults
    model, saved = train_with_sb3(
        total_timesteps= 10_000, #500_000,
        outdir="runs/sb3_boxing",
        n_envs=4,
        eval_freq= 500, #50_000,
        n_eval_episodes=5,
        device="auto"
    )
    print("Done.")

    from plot_training import plot_from_run
    plot_from_run("runs/sb3_boxing")