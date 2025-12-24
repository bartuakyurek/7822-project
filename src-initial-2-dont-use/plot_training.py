import os
import json
import matplotlib.pyplot as plt

def load_eval_results(directory):
    eval_files = [f for f in os.listdir(directory) if f.startswith("eval_") and f.endswith(".json")]
    eval_files = sorted(eval_files, key=lambda fn: int(fn.split("_")[1].split(".")[0]))
    
    timesteps = []
    mean_rewards = []
    
    for fname in eval_files:
        with open(os.path.join(directory, fname), "r") as f:
            data = json.load(f)
            timesteps.append(data["timesteps"])
            mean_rewards.append(data["mean_reward"])
    
    return timesteps, mean_rewards

def plot_training_curve(timesteps, mean_rewards, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_rewards, marker='o')
    plt.title("SB3 PPO Training — Mean Reward Over Time")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curve to: {save_path}")
    else:
        plt.show()

def plot_from_run(run_dir):
    timesteps, rewards = load_eval_results(run_dir)
    plot_training_curve(timesteps, rewards,
                        save_path=os.path.join(run_dir, "training_curve.png"))

if __name__ == "__main__":
    plot_from_run("runs/sb3_boxing")