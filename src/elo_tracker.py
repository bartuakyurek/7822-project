# plot_eval_results.py
import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# elo_tracker.py
import json
import os

class EloTracker:
    def __init__(self, initial_elo=1200, k=32):
        self.elo = initial_elo
        self.k = k
        self.elo_history = []  # (timestep, elo) pairs
        self.win_history = []  # (timestep, win) pairs
        self.timesteps = []
        
    def update(self, win: bool, timestep: int = None):
        expected_score = 1 / (1 + 10 ** ((1000 - self.elo) / 400))
        score = 1 if win else 0
        self.elo += self.k * (score - expected_score)
        
        # Record history
        if timestep is not None:
            self.elo_history.append((timestep, self.elo))
            self.win_history.append((timestep, score))
            self.timesteps.append(timestep)
        
        return self.elo
    
    def get_win_rate(self, window=10):
        """Calculate win rate over last N games"""
        if len(self.win_history) == 0:
            return 0.0
        recent_wins = [w for _, w in self.win_history[-window:]]
        return sum(recent_wins) / len(recent_wins)
    
    def save(self, filepath):
        """Save tracker state to JSON"""
        data = {
            'current_elo': self.elo,
            'k': self.k,
            'elo_history': self.elo_history,
            'win_history': self.win_history,
            'timesteps': self.timesteps
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load tracker state from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check if this is an ELO tracker file or an eval stats file
        if 'current_elo' not in data:
            raise ValueError(f"File {filepath} is not an ELO tracker save file. "
                           f"Use 'elo_tracker.json' or 'elo_tracker_final.json' instead.")
        
        tracker = cls(initial_elo=data['current_elo'], k=data['k'])
        tracker.elo_history = [tuple(x) for x in data['elo_history']]
        tracker.win_history = [tuple(x) for x in data['win_history']]
        tracker.timesteps = data['timesteps']
        return tracker


def plot_elo_and_winrate(elo_tracker, save_dir=None):
    """Create two plots: ELO over time and win rate over time"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if len(elo_tracker.timesteps) == 0:
        print("No data to plot")
        return
    
    # Extract data
    timesteps = elo_tracker.timesteps
    elos = [elo for _, elo in elo_tracker.elo_history]
    
    # Calculate rolling win rate (window of 10 games)
    window = 10
    win_rates = []
    win_rate_timesteps = []
    for i in range(len(elo_tracker.win_history)):
        start_idx = max(0, i - window + 1)
        recent_wins = [w for _, w in elo_tracker.win_history[start_idx:i+1]]
        win_rate = sum(recent_wins) / len(recent_wins)
        win_rates.append(win_rate)
        win_rate_timesteps.append(elo_tracker.win_history[i][0])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot ELO
    ax1.plot(timesteps, elos, linewidth=2)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('ELO Rating')
    ax1.set_title('ELO Rating vs. Training Timesteps')
    ax1.grid(True, alpha=0.3)
    
    # Plot Win Rate
    ax2.plot(win_rate_timesteps, win_rates, linewidth=2, color='green')
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Win Rate')
    ax2.set_title(f'Win Rate vs. Training Timesteps (Rolling Window = {window})')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'elo_and_winrate.png'), dpi=150)
        print(f"Saved plot to {os.path.join(save_dir, 'elo_and_winrate.png')}")
    
    plt.show()


def plot_from_run_dir(run_dir):
    """Load ELO tracker from a run directory and plot"""
    # Try both possible filenames
    for filename in ['elo_tracker_final.json', 'elo_tracker.json']:
        filepath = os.path.join(run_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading ELO tracker from {filepath}")
            tracker = EloTracker.load(filepath)
            plot_elo_and_winrate(tracker, save_dir=run_dir)
            return tracker
    
    print(f"No ELO tracker file found in {run_dir}")
    print("Looking for: elo_tracker.json or elo_tracker_final.json")
    return None


def load_all_eval_files(run_dir):
    """Load all evaluation JSON files from a run directory"""
    eval_files = glob.glob(os.path.join(run_dir, "eval_*.json"))
    
    # Sort by timestep number extracted from filename
    def get_timestep(filepath):
        filename = os.path.basename(filepath)
        # Extract number from "eval_1000.json"
        timestep = int(filename.replace("eval_", "").replace(".json", ""))
        return timestep
    
    eval_files = sorted(eval_files, key=get_timestep)
    
    data = []
    for filepath in eval_files:
        with open(filepath, 'r') as f:
            eval_data = json.load(f)
            data.append(eval_data)
    
    return data

def plot_from_eval_files(run_dir, save_plot=True):
    """
    Plot ELO and win rates from evaluation JSON files.
    This is useful if training was interrupted and elo_tracker.json wasn't saved.
    """
    eval_data = load_all_eval_files(run_dir)
    
    if not eval_data:
        print(f"No evaluation files found in {run_dir}")
        return
    
    # Extract data
    timesteps = []
    elos = []
    win_rates = []
    mean_rewards = []
    
    for entry in eval_data:
        timesteps.append(entry['timesteps'])
        if 'elo' in entry and entry['elo'] is not None:
            elos.append(entry['elo'])
        if 'win_rate' in entry:
            win_rates.append(entry['win_rate'])
        mean_rewards.append(entry['mean_reward'])
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: ELO (if available)
    if elos and len(elos) == len(timesteps):
        axes[0].plot(timesteps, elos, linewidth=2, marker='o')
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('ELO Rating')
        axes[0].set_title('ELO Rating Over Training')
        axes[0].grid(True, alpha=0.3)
    else:
        # Plot mean reward instead if no ELO
        axes[0].plot(timesteps, mean_rewards, linewidth=2, marker='o', color='orange')
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Mean Reward Over Training')
        axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Win Rate
    if win_rates and len(win_rates) == len(timesteps):
        axes[1].plot(timesteps, win_rates, linewidth=2, marker='o', color='green')
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel('Win Rate')
        axes[1].set_title('Win Rate Over Training')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        output_path = os.path.join(run_dir, 'eval_results.png')
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Summary from {len(eval_data)} evaluation points:")
    print(f"{'='*50}")
    print(f"Timesteps range: {timesteps[0]} - {timesteps[-1]}")
    if elos:
        print(f"ELO: {elos[0]:.1f} -> {elos[-1]:.1f}")
    if win_rates:
        print(f"Win rate: {win_rates[0]:.2%} -> {win_rates[-1]:.2%}")
    print(f"Mean reward: {mean_rewards[0]:.2f} -> {mean_rewards[-1]:.2f}")
    print(f"{'='*50}\n")

def inspect_eval_file(filepath):
    """Print contents of a single evaluation file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"\nFile: {filepath}")
    print(f"{'='*50}")
    for key, value in data.items():
        if key == 'rewards':
            print(f"{key}: {value[:3]}... (showing first 3)")
        else:
            print(f"{key}: {value}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # Example usage
    run_dir = "runs/sb3_boxing"
    
    # Option 1: Plot all evaluation data
    plot_from_eval_files(run_dir)
    
    # Option 2: Inspect a specific evaluation file
    # inspect_eval_file("runs/sb3_boxing/eval_500.json")
