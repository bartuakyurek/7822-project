
import gymnasium as gym
import numpy as np
import torch
import imageio
from datetime import datetime
import os

from training import FrameStack
from training import load_policy

def preprocess_frame(frame):
    """Preprocess frame: grayscale and resize"""
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    from scipy.ndimage import zoom
    resized = zoom(gray, (84/210, 84/160), order=1)
    return resized.astype(np.uint8)



def render_episode(policy, env, frame_stack, device='cpu', 
                   deterministic=True, max_steps=5000):
    """
    Render a single episode and return frames
    
    Args:
        policy: Trained policy network
        env: Gymnasium environment
        frame_stack: FrameStack object
        device: torch device
        deterministic: Use deterministic actions (greedy)
        max_steps: Maximum steps per episode
    
    Returns:
        frames: List of RGB frames
        total_reward: Episode reward
        episode_length: Number of steps
    """
    frames = []
    obs, info = env.reset()
    
    # Store original RGB frame
    frames.append(obs)
    
    # Preprocess for policy
    processed = preprocess_frame(obs)
    state = frame_stack.reset(processed)
    
    total_reward = 0
    episode_length = 0
    
    for step in range(max_steps):
        # Get action from policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action, _, _ = policy.get_action(state_tensor, deterministic=deterministic)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # Store RGB frame
        frames.append(obs)
        
        # Process for policy
        processed = preprocess_frame(obs)
        state = frame_stack.step(processed)
        
        total_reward += reward
        episode_length += 1
        
        if done:
            break
    
    return frames, total_reward, episode_length

def create_video(frames, output_path, fps=30, add_text=True, 
                 reward=None, episode_length=None):
    """
    Create video from frames
    
    Args:
        frames: List of RGB frames
        output_path: Path to save video
        fps: Frames per second
        add_text: Add text overlay with stats
        reward: Episode reward to display
        episode_length: Episode length to display
    """
    if add_text and reward is not None:
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Add text overlay to frames
            annotated_frames = []
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                
                # Try to use a nice font, fall back to default
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # Add text
                text = f"Step: {i}/{episode_length}  Reward: {reward:.1f}"
                
                # Add black background for text
                bbox = draw.textbbox((10, 10), text, font=font)
                draw.rectangle(bbox, fill='black')
                draw.text((10, 10), text, fill='white', font=font)
                
                annotated_frames.append(np.array(img))
            
            frames = annotated_frames
        except ImportError:
            print("PIL not available, saving video without text overlay")
    
    # Save video
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"✓ Video saved to {output_path}")

def render_multiple_episodes(checkpoint_path, n_episodes=5, 
                            output_dir='videos', deterministic=True,
                            fps=30, device='cpu'):
    """
    Render multiple episodes and save as videos
    
    Args:
        checkpoint_path: Path to checkpoint file
        n_episodes: Number of episodes to render
        output_dir: Directory to save videos
        deterministic: Use deterministic actions
        fps: Video frame rate
        device: torch device
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment with rendering
    env = gym.make('ALE/Boxing-v5', render_mode='rgb_array')
    n_actions = env.action_space.n
    
    # Load policy
    policy = load_policy(checkpoint_path, n_actions, device)
    
    # Create frame stack
    frame_stack = FrameStack(4)
    
    print(f"\nRendering {n_episodes} episodes...")
    print("="*60)
    
    all_rewards = []
    all_lengths = []
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        # Reset frame stack for new episode
        frame_stack = FrameStack(4)
        
        # Render episode
        frames, reward, length = render_episode(
            policy, env, frame_stack, device, 
            deterministic=deterministic
        )
        
        all_rewards.append(reward)
        all_lengths.append(length)
        
        print(f"  Reward: {reward:.2f}")
        print(f"  Length: {length} steps")
        
        # Save video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = f'boxing_episode_{episode+1}_{timestamp}.mp4'
        video_path = os.path.join(output_dir, video_name)
        
        create_video(frames, video_path, fps=fps, add_text=True,
                    reward=reward, episode_length=length)
    
    env.close()
    
    # Print summary
    print("\n" + "="*60)
    print("RENDERING SUMMARY")
    print("="*60)
    print(f"Episodes rendered: {n_episodes}")
    print(f"Average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f} steps")
    print(f"Best reward: {max(all_rewards):.2f}")
    print(f"Worst reward: {min(all_rewards):.2f}")
    print(f"\nVideos saved in: {output_dir}/")
    print("="*60)

def create_comparison_video(checkpoint_paths, labels, output_path='videos/comparison.mp4',
                           fps=30, device='cpu'):
    """
    Create a side-by-side comparison video of multiple policies
    
    Args:
        checkpoint_paths: List of checkpoint paths
        labels: List of labels for each policy
        output_path: Path to save comparison video
        fps: Video frame rate
        device: torch device
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Create environments
    envs = [gym.make('ALE/Boxing-v5', render_mode='rgb_array') for _ in checkpoint_paths]
    n_actions = envs[0].action_space.n
    
    # Load policies
    policies = [load_policy(cp, n_actions, device) for cp in checkpoint_paths]
    
    # Create frame stacks
    frame_stacks = [FrameStack(4) for _ in checkpoint_paths]
    
    # Reset all environments
    states = []
    for i, env in enumerate(envs):
        obs, _ = env.reset()
        processed = preprocess_frame(obs)
        states.append(frame_stacks[i].reset(processed))
    
    frames = []
    done_flags = [False] * len(envs)
    rewards = [0] * len(envs)
    
    max_steps = 5000
    for step in range(max_steps):
        # Collect frames from all environments
        env_frames = []
        
        for i, (env, policy, frame_stack, state) in enumerate(zip(envs, policies, frame_stacks, states)):
            if not done_flags[i]:
                # Get action
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _ = policy.get_action(state_tensor, deterministic=True)
                
                # Step
                obs, reward, terminated, truncated, info = env.step(action.item())
                rewards[i] += reward
                done_flags[i] = terminated or truncated
                
                # Update state
                processed = preprocess_frame(obs)
                states[i] = frame_stack.step(processed)
            else:
                obs = envs[i].render()
            
            # Add label to frame
            img = Image.fromarray(obs)
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            text = f"{labels[i]}: {rewards[i]:.1f}"
            bbox = draw.textbbox((5, 5), text, font=font)
            draw.rectangle(bbox, fill='black')
            draw.text((5, 5), text, fill='white', font=font)
            
            env_frames.append(np.array(img))
        
        # Concatenate frames side by side
        if len(env_frames) == 2:
            combined = np.concatenate(env_frames, axis=1)
        else:
            # Stack in grid for more than 2
            combined = np.concatenate(env_frames, axis=1)
        
        frames.append(combined)
        
        if all(done_flags):
            break
    
    # Close environments
    for env in envs:
        env.close()
    
    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"\n✓ Comparison video saved to {output_path}")
    print(f"Final rewards: {dict(zip(labels, rewards))}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Render videos of trained Boxing agent')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/boxing_best_policy.pth',
                       help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to render')
    parser.add_argument('--output-dir', type=str, default='videos',
                       help='Directory to save videos')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video frame rate')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic actions instead of deterministic')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Compare multiple checkpoints (space-separated paths)')
    parser.add_argument('--labels', nargs='+', type=str,
                       help='Labels for comparison (space-separated)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Boxing Agent Video Renderer")
    print("="*60)
    print("\nRequired packages:")
    print("pip install gymnasium[atari] ale-py torch numpy scipy imageio[ffmpeg] pillow")
    print("="*60)
    
    if args.compare:
        # Comparison mode
        if not args.labels:
            args.labels = [f"Policy {i+1}" for i in range(len(args.compare))]
        
        print(f"\nComparing {len(args.compare)} policies:")
        for cp, label in zip(args.compare, args.labels):
            print(f"  • {label}: {cp}")
        
        create_comparison_video(
            args.compare, 
            args.labels,
            output_path=os.path.join(args.output_dir, 'comparison.mp4'),
            fps=args.fps,
            device=args.device
        )
    else:
        # Single policy mode
        print(f"\nRendering {args.episodes} episodes from: {args.checkpoint}")
        print(f"Mode: {'Stochastic' if args.stochastic else 'Deterministic'}")
        
        render_multiple_episodes(
            checkpoint_path=args.checkpoint,
            n_episodes=args.episodes,
            output_dir=args.output_dir,
            deterministic=not args.stochastic,
            fps=args.fps,
            device=args.device
        )
    
    print("\n✨ Done! Check the videos in the output directory.")