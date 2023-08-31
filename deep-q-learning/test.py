import torch
import gymnasium as gym
from dqn import QAgent
from gymnasium.wrappers import record_video

env = gym.make('Taxi-v3', render_mode="rgb_array")
env = record_video.RecordVideo(env, video_folder='runs')
model = torch.load('.models/pytorch_1680419654.5881696.pt')

state, _ = env.reset()
done = False
env.render()

while not done:
    action = model.target_net(torch.tensor([state], device=model.device)).max(1)[1].item()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state
