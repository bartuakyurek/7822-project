# train_selfplay.py
"""
Self-play + PPO for PettingZoo Boxing.
- Uses Stable-Baselines3 PPO (single policy, parameter sharing).
- Periodically saves checkpoints to opponent pool.
- Periodically runs evaluation matches (n_eval_games) vs opponents in pool and updates Elo.
"""

import os
import time
import random
import argparse
import json
from collections import defaultdict
import numpy as np

# RL libs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# PettingZoo / wrappers
from pettingzoo.atari import boxing_v2
from pettingzoo.utils.wrappers import GymWrapper
import gymnasium as gym

# Elo helper
from elo import expected_score, update_elo

# --- Helpers to wrap PettingZoo for SB3 parameter-shared policy ---
def make_sb3_env(seed=None):
    """
    Create a gym-style environment where the single SB3 agent controls both players by stacking their observations.
    Simpler approach: control only player_0 with the policy and use a fixed opponent policy (from pool) to provide actions for player_1.
    For parameter-sharing self-play: we will create a wrapper class that at each step obtains actions for both agents from the same policy.
    For clarity in this script, we'll use GymWrapper to produce a single-agent gym env for player 0 and supply opponent actions externally in the self-play loop.
    """
    env = boxing_v2.env()
    env.reset()
    # We'll control player_0 only with SB3; opponent actions will be injected from a provided policy.
    # Use GymWrapper to expose single-player gym interface for a single agent.
    gym_env = GymWrapper(env)  # this yields a gym-like env but note: will need careful stepping logic for multi-agent.
    return gym_env

# --- Simple evaluation vs a saved opponent model file ---
def load_policy_for_inference(path):
    """Return a callable that given obs returns an action. For SB3 policies saved with model.save()."""
    model = PPO.load(path)
    def act(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return act

def play_match(env, agent_act_fn, opponent_act_fn, max_steps=1200):
    """
    Plays one match in the PettingZoo boxing env.
    agent_act_fn(observation) -> action
    opponent_act_fn(observation) -> action
    Returns: (agent_reward_total, opponent_reward_total, winner) where winner in {agent, opponent, draw}
    """
    # We need to use the underlying PettingZoo env for multi-agent stepping.
    # Use environment reset/step via original env (not GymWrapper).
    pet_env = boxing_v3.env()
    obs = pet_env.reset()
    # obs is a dict mapping agent -> observation (or AEC API iteration). For boxing_v3 the agent order is ['player_0','player_1']
    total = { 'player_0': 0.0, 'player_1': 0.0 }
    for step in range(max_steps):
        # In PettingZoo AEC, agents are stepped sequentially; easiest is to use the parallel API if available.
        # The boxing_v3 supports the AEC API: use pet_env.last() pattern; but to keep it robust we'll use the parallel_env wrapper:
        break

    # Instead, create parallel env
    pet_env = boxing_v3.parallel_env()
    obs = pet_env.reset()
    done = {'player_0': False, 'player_1': False}
    infos = {}
    for t in range(max_steps):
        a0 = agent_act_fn(obs['player_0'])
        a1 = opponent_act_fn(obs['player_1'])
        actions = {'player_0': a0, 'player_1': a1}
        obs, rewards, terminated, truncated, infos = pet_env.step(actions)
        total['player_0'] += rewards['player_0']
        total['player_1'] += rewards['player_1']
        if any(terminated.values()) or any(truncated.values()):
            break

    # Determine winner by game rules: higher points or KO. We'll judge by cumulative reward.
    if total['player_0'] > total['player_1']:
        winner = 'agent'
    elif total['player_0'] < total['player_1']:
        winner = 'opponent'
    else:
        winner = 'draw'

    return total['player_0'], total['player_1'], winner

# --- Main training loop ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="runs/boxing_selfplay")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint_interval", type=int, default=100_000)
    parser.add_argument("--eval_interval", type=int, default=100_000)
    parser.add_argument("--n_eval_games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()

class EvalCallback(BaseCallback):
    def __init__(self, outdir, evaluator_fn, eval_interval, n_eval_games, verbose=1):
        super().__init__(verbose)
        self.outdir = outdir
        self.evaluator_fn = evaluator_fn
        self.eval_interval = eval_interval
        self.n_eval_games = n_eval_games
        self.last_eval = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_eval) >= self.eval_interval:
            self.last_eval = self.num_timesteps
            self.evaluator_fn(self.num_timesteps, n_games=self.n_eval_games)
        return True

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create a single-agent gym env (we will control player_0). Using vector env for SB3.
    def make_env():
        env = boxing_v3.env()
        # We will use the parallel API for evaluation, but here we use a custom lightweight wrapper for SB3: control only 'player_0'.
        # PettingZoo's GymWrapper maps the multi-agent env to gym by controlling a single agent; but for robust usage consider the tutorial:
        from pettingzoo.utils.wrappers import BaseParallelWrapper
        # Use GymWrapper for demonstration; you may need to adjust observation/action spaces depending on SB3 expectations.
        from pettingzoo.utils.wrappers import GymWrapper
        return GymWrapper(env)

    vec_env = DummyVecEnv([make_env])
    # Optionally normalize
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model = PPO("CnnPolicy", vec_env, verbose=1)  # boxing is image-based -> use CnnPolicy
    # Checkpointing
    checkpoint_cb = CheckpointCallback(save_freq=args.checkpoint_interval//1000, save_path=args.outdir, name_prefix="ppo_boxing")
    # We'll implement evaluation function that saves checkpoint then runs evaluation vs pool and updates Elo.
    opponent_pool = []  # list of saved model paths
    elo_scores = {}     # mapping model_path -> rating
    rating_history = []

    def evaluator(timesteps, n_games=20):
        # Save current model
        ckpt_path = os.path.join(args.outdir, f"ckpt_{timesteps}.zip")
        model.save(ckpt_path)
        opponent_pool.append(ckpt_path)
        if ckpt_path not in elo_scores:
            elo_scores[ckpt_path] = 1500.0

        # Evaluate current checkpoint vs random opponents from pool (excluding itself),
        # and update Elo based on match outcomes.
        results = defaultdict(int)  # wins for current
        matches = []
        # select opponents: sample up to m opponents
        opponents = random.sample(opponent_pool, min(5, max(1, len(opponent_pool))))
        # ensure at least one older opponent
        opponents = [o for o in opponents if o != ckpt_path]
        opponents = opponents or [ckpt_path]  # fallback

        agent_fn = lambda obs: model.predict(obs, deterministic=True)[0]

        for opp_path in opponents:
            # load opponent policy
            opp_model = PPO.load(opp_path)
            opp_fn = lambda obs: opp_model.predict(obs, deterministic=True)[0]

            wins = 0
            losses = 0
            draws = 0
            for i in range(n_games):
                a_r, o_r, winner = play_match(None, agent_fn, opp_fn)
                if winner == 'agent':
                    wins += 1
                elif winner == 'opponent':
                    losses += 1
                else:
                    draws += 1

                # Elo: update per-game (use 1/0/0.5)
                score = 1.0 if winner == 'agent' else (0.5 if winner == 'draw' else 0.0)
                ra = elo_scores.get(ckpt_path,1500.0)
                rb = elo_scores.get(opp_path,1500.0)
                na, nb = update_elo(ra, rb, score, k=20)
                elo_scores[ckpt_path] = na
                elo_scores[opp_path] = nb

            results[opp_path] = (wins, losses, draws)

        # write results + Elo snapshot
        snapshot = {
            "timesteps": timesteps,
            "elo": dict(elo_scores),
            "results": {k: v for k,v in results.items()}
        }
        fname = os.path.join(args.outdir, f"eval_{timesteps}.json")
        with open(fname, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"Saved evaluation snapshot to {fname}")

    eval_cb = EvalCallback(args.outdir, evaluator, eval_interval=args.eval_interval, n_eval_games=args.n_eval_games)
    # combine callbacks (we can pass list to model.learn via callback argument)
    model.learn(total_timesteps=args.total_timesteps, callback=[checkpoint_cb, eval_cb])

if __name__ == "__main__":
    main()
