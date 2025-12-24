import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os
from datetime import datetime

class ELOTracker:
    """Track ELO ratings for self-play training"""
    def __init__(self, k_factor=32, initial_elo=1500):
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        
        # Track ELO over time
        self.elo_history = []
        self.current_elo = initial_elo
        self.opponent_elo = initial_elo
        
        # Track match results
        self.match_results = []
        self.win_rate_history = []
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score using ELO formula"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_elo(self, player_reward, opponent_reward):
        """
        Update ELO based on match outcome.
        player_reward > opponent_reward = win (score=1)
        player_reward = opponent_reward = draw (score=0.5)
        player_reward < opponent_reward = loss (score=0)
        """
        # Determine match outcome
        if player_reward > opponent_reward:
            actual_score = 1.0  # Win
            result = 'win'
        elif player_reward < opponent_reward:
            actual_score = 0.0  # Loss
            result = 'loss'
        else:
            actual_score = 0.5  # Draw
            result = 'draw'
        
        # Calculate expected score
        expected = self.expected_score(self.current_elo, self.opponent_elo)
        
        # Update ELO
        elo_change = self.k_factor * (actual_score - expected)
        self.current_elo += elo_change
        
        # Store history
        self.elo_history.append(self.current_elo)
        self.match_results.append({
            'result': result,
            'player_reward': player_reward,
            'opponent_reward': opponent_reward,
            'elo': self.current_elo,
            'elo_change': elo_change
        })
        
        return elo_change, result
    
    def update_opponent_elo(self):
        """Update opponent ELO to current player ELO (when swapping opponent)"""
        self.opponent_elo = self.current_elo
    
    def add_episode_data(self, reward, length):
        """Store episode statistics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Calculate rolling win rate (last 50 matches)
        recent_results = self.match_results[-50:]
        if recent_results:
            wins = sum(1 for r in recent_results if r['result'] == 'win')
            win_rate = wins / len(recent_results)
            self.win_rate_history.append(win_rate)
        else:
            self.win_rate_history.append(0.5)
    
    def get_stats(self, window=50):
        """Get recent statistics"""
        recent_rewards = self.episode_rewards[-window:]
        recent_results = self.match_results[-window:]
        
        stats = {
            'current_elo': self.current_elo,
            'avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'avg_length': np.mean(self.episode_lengths[-window:]) if self.episode_lengths else 0,
            'win_rate': sum(1 for r in recent_results if r['result'] == 'win') / len(recent_results) if recent_results else 0,
            'total_episodes': len(self.episode_rewards)
        }
        return stats
    
    def save(self, filepath='elo_tracker.json'):
        """Save tracker state to file"""
        data = {
            'elo_history': self.elo_history,
            'match_results': self.match_results,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'win_rate_history': self.win_rate_history,
            'current_elo': self.current_elo,
            'opponent_elo': self.opponent_elo
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath='elo_tracker.json'):
        """Load tracker state from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.elo_history = data['elo_history']
        self.match_results = data['match_results']
        self.episode_rewards = data['episode_rewards']
        self.episode_lengths = data['episode_lengths']
        self.win_rate_history = data['win_rate_history']
        self.current_elo = data['current_elo']
        self.opponent_elo = data['opponent_elo']

def plot_training_progress(elo_tracker, save_dir='training_plots'):
    """Generate comprehensive training plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Boxing Self-Play Training Progress', fontsize=16, fontweight='bold')
    
    # 1. ELO Rating over time
    ax1 = axes[0, 0]
    if elo_tracker.elo_history:
        ax1.plot(elo_tracker.elo_history, linewidth=2, color='#2E86AB')
        ax1.axhline(y=elo_tracker.initial_elo, color='red', linestyle='--', 
                   alpha=0.5, label='Initial ELO')
        ax1.fill_between(range(len(elo_tracker.elo_history)), 
                        elo_tracker.initial_elo, 
                        elo_tracker.elo_history, 
                        alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('ELO Rating', fontsize=11)
        ax1.set_title('ELO Rating Progression', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. Win Rate over time
    ax2 = axes[0, 1]
    if elo_tracker.win_rate_history:
        window = 20
        smoothed_wr = np.convolve(elo_tracker.win_rate_history, 
                                  np.ones(window)/window, mode='valid')
        ax2.plot(elo_tracker.win_rate_history, alpha=0.3, color='gray', label='Raw')
        ax2.plot(range(window-1, len(elo_tracker.win_rate_history)), 
                smoothed_wr, linewidth=2, color='#A23B72', label=f'{window}-Episode MA')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Win Rate')
        ax2.set_xlabel('Episode', fontsize=11)
        ax2.set_ylabel('Win Rate', fontsize=11)
        ax2.set_title('Win Rate (Rolling)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. Episode Rewards
    ax3 = axes[0, 2]
    if elo_tracker.episode_rewards:
        window = 50
        rewards = np.array(elo_tracker.episode_rewards)
        ax3.plot(rewards, alpha=0.2, color='gray')
        
        # Moving average
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(rewards)), ma, 
                    linewidth=2, color='#F18F01', label=f'{window}-Episode MA')
        
        ax3.set_xlabel('Episode', fontsize=11)
        ax3.set_ylabel('Reward', fontsize=11)
        ax3.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Match Outcome Distribution
    ax4 = axes[1, 0]
    if elo_tracker.match_results:
        results = [r['result'] for r in elo_tracker.match_results]
        unique, counts = np.unique(results, return_counts=True)
        colors = {'win': '#06A77D', 'loss': '#D62246', 'draw': '#F77F00'}
        bar_colors = [colors.get(r, 'gray') for r in unique]
        
        bars = ax4.bar(unique, counts, color=bar_colors, alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Match Outcome Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({100*count/total:.1f}%)',
                    ha='center', va='bottom', fontsize=10)
    
    # 5. ELO Change Distribution
    ax5 = axes[1, 1]
    if elo_tracker.match_results:
        elo_changes = [r['elo_change'] for r in elo_tracker.match_results]
        ax5.hist(elo_changes, bins=30, color='#6A4C93', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
        ax5.set_xlabel('ELO Change', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('ELO Change Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.legend()
    
    # 6. Episode Length over time
    ax6 = axes[1, 2]
    if elo_tracker.episode_lengths:
        window = 50
        lengths = np.array(elo_tracker.episode_lengths)
        ax6.plot(lengths, alpha=0.2, color='gray')
        
        if len(lengths) >= window:
            ma = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax6.plot(range(window-1, len(lengths)), ma, 
                    linewidth=2, color='#9D4EDD', label=f'{window}-Episode MA')
        
        ax6.set_xlabel('Episode', fontsize=11)
        ax6.set_ylabel('Steps', fontsize=11)
        ax6.set_title('Episode Length', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(save_dir, f'training_progress_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    
    return fig

def plot_elo_comparison(elo_trackers, labels, save_dir='training_plots'):
    """Compare ELO progression across multiple training runs"""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62246']
    
    for i, (tracker, label) in enumerate(zip(elo_trackers, labels)):
        color = colors[i % len(colors)]
        if tracker.elo_history:
            plt.plot(tracker.elo_history, linewidth=2, color=color, 
                    label=label, alpha=0.8)
    
    plt.axhline(y=1500, color='black', linestyle='--', alpha=0.3, label='Initial ELO')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('ELO Rating', fontsize=12)
    plt.title('ELO Comparison Across Training Runs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(save_dir, f'elo_comparison_{timestamp}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {filepath}")
    
    plt.close()

def print_training_summary(elo_tracker, window=50):
    """Print detailed training summary"""
    stats = elo_tracker.get_stats(window)
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Episodes:        {stats['total_episodes']}")
    print(f"Current ELO:           {stats['current_elo']:.1f}")
    print(f"ELO Gain:              {stats['current_elo'] - elo_tracker.initial_elo:+.1f}")
    print(f"Recent Win Rate:       {stats['win_rate']*100:.1f}%")
    print(f"Avg Reward (last {window}):  {stats['avg_reward']:.2f}")
    print(f"Avg Length (last {window}):  {stats['avg_length']:.1f} steps")
    
    if elo_tracker.match_results:
        recent = elo_tracker.match_results[-window:]
        wins = sum(1 for r in recent if r['result'] == 'win')
        losses = sum(1 for r in recent if r['result'] == 'loss')
        draws = sum(1 for r in recent if r['result'] == 'draw')
        
        print(f"\nRecent Results (last {window}):")
        print(f"  Wins:   {wins} ({wins/len(recent)*100:.1f}%)")
        print(f"  Losses: {losses} ({losses/len(recent)*100:.1f}%)")
        print(f"  Draws:  {draws} ({draws/len(recent)*100:.1f}%)")
    
    if elo_tracker.elo_history:
        max_elo = max(elo_tracker.elo_history)
        max_episode = elo_tracker.elo_history.index(max_elo)
        print(f"\nPeak ELO: {max_elo:.1f} (Episode {max_episode})")
    
    print("="*60 + "\n")

# Example usage function
def demo_tracking():
    """Demonstrate ELO tracking and plotting"""
    # Create tracker
    tracker = ELOTracker(k_factor=32, initial_elo=1500)
    
    # Simulate some training data
    np.random.seed(42)
    for episode in range(500):
        # Simulate improving performance
        skill = episode / 500
        player_reward = np.random.normal(skill * 20 - 10, 5)
        opponent_reward = np.random.normal(0, 5)
        episode_length = int(np.random.normal(500, 100))
        
        # Update tracker
        tracker.update_elo(player_reward, opponent_reward)
        tracker.add_episode_data(player_reward, episode_length)
        
        # Periodically update opponent
        if (episode + 1) % 50 == 0:
            tracker.update_opponent_elo()
    
    # Generate plots
    plot_training_progress(tracker)
    
    # Print summary
    print_training_summary(tracker)
    
    # Save tracker
    tracker.save('demo_elo_tracker.json')
    
    print("Demo complete! Check the 'training_plots' directory for visualizations.")

if __name__ == "__main__":
    print("ELO Tracker Demo")
    print("This demonstrates the tracking and plotting functionality.")
    print()
    demo_tracking()