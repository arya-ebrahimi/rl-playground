import gymnasium as gym
import numpy as np
from gymnasium.wrappers import record_video

from core.policy_iteration import policy_iteration
from core.value_iteration import value_iteration

class Wrapper(gym.Wrapper):
    def __init__(self, env, step_reward=0.0, hole_reward=0.0):
        super().__init__(env)
    
        self.step_reward = step_reward
        self.hole_reward = hole_reward
        
        self.holes = [5, 7, 11, 12]
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward = self.step_reward
        if observation in self.holes:
            reward = self.hole_reward
        if observation == 15:
            reward = 1.0
        return observation, reward,  terminated, truncated, info
    
def play(mode, gamma=0.9, is_slippery=False, step_reward=0.0, hole_reward=0.0):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode='rgb_array')
    env = Wrapper(env=env, step_reward=step_reward, hole_reward=hole_reward)
    name = mode+',gamma:'+str(gamma)+',is_slippery:'+str(is_slippery)+',step_reward:'+str(step_reward)+',hole_reward:'+str(hole_reward)
    env = record_video.RecordVideo(env, video_folder='runs', name_prefix=name)
    if mode == 'policy_iteration':
        V, pi = policy_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)
    else:
        V, pi = value_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=gamma, tol=1e-4)
        
    done = False
    state, _ = env.reset()
    env.render()
    total_reward = 0
    steps = 0
    while not done:
        steps+=1
        action = np.argmax(pi[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        # env.render()
    env.close()
    print('total reward is: ', total_reward)
    
    return name, steps, total_reward
    
if __name__ == '__main__':
    total_steps = []
    total_steps.append(play(mode='policy_iteration', gamma=0.0))
    total_steps.append(play(mode='value_iteration', gamma=0.0))
    total_steps.append(play(mode='policy_iteration', gamma=0.9))
    total_steps.append(play(mode='value_iteration', gamma=0.9))
    total_steps.append(play(mode='policy_iteration', gamma=1.0))
    total_steps.append(play(mode='value_iteration', gamma=1.0))
    total_steps.append(play(mode='policy_iteration', gamma=0.9, is_slippery=True))
    total_steps.append(play(mode='value_iteration', gamma=0.9, is_slippery=True))
    
    total_steps.append(play(mode='policy_iteration', gamma=0.9, is_slippery=False, step_reward=-0.05))
    total_steps.append(play(mode='value_iteration', gamma=0.9, is_slippery=False, step_reward=-0.05))
    
    total_steps.append(play(mode='policy_iteration', gamma=1, is_slippery=False, step_reward=-0.05))
    total_steps.append(play(mode='value_iteration', gamma=1, is_slippery=False, step_reward=-0.05))
    
    total_steps.append(play(mode='policy_iteration', gamma=0.9, is_slippery=True, step_reward=-0.05))
    total_steps.append(play(mode='value_iteration', gamma=0.9, is_slippery=True, step_reward=-0.05))

    total_steps.append(play(mode='policy_iteration', gamma=1, is_slippery=True, step_reward=-0.05))
    total_steps.append(play(mode='value_iteration', gamma=1, is_slippery=True, step_reward=-0.05))
    
    total_steps.append(play(mode='policy_iteration', gamma=0.9, is_slippery=False, hole_reward=-2.0))
    total_steps.append(play(mode='value_iteration', gamma=0.9, is_slippery=False, hole_reward=-2.0))
    
    total_steps.append(play(mode='policy_iteration', gamma=1, is_slippery=False, hole_reward=-2.0))
    total_steps.append(play(mode='value_iteration', gamma=1, is_slippery=False, hole_reward=-2.0))
        
    total_steps.append(play(mode='policy_iteration', gamma=0.9, is_slippery=True, hole_reward=-2.0))
    total_steps.append(play(mode='value_iteration', gamma=0.9, is_slippery=True, hole_reward=-2.0))
    
    total_steps.append(play(mode='policy_iteration', gamma=1, is_slippery=True, hole_reward=-2.0))
    total_steps.append(play(mode='value_iteration', gamma=1, is_slippery=True, hole_reward=-2.0))

    
    total_steps.append(play(mode='policy_iteration', gamma=0.9, is_slippery=False, step_reward=-0.05, hole_reward=-2.0))
    total_steps.append(play(mode='value_iteration', gamma=0.9, is_slippery=False, step_reward=-0.05, hole_reward=-2.0))
    
        
    total_steps.append(play(mode='policy_iteration', gamma=1, is_slippery=False, step_reward=-0.05, hole_reward=-2.0))
    total_steps.append(play(mode='value_iteration', gamma=1, is_slippery=False, step_reward=-0.05, hole_reward=-2.0))
    

    
    total_steps.append(play(mode='policy_iteration', gamma=0.9, is_slippery=True, step_reward=-0.05, hole_reward=-2.0))
    total_steps.append(play(mode='value_iteration', gamma=0.9, is_slippery=True, step_reward=-0.05, hole_reward=-2.0))
    
    total_steps.append(play(mode='policy_iteration', gamma=1, is_slippery=True, step_reward=-0.05, hole_reward=-2.0))
    total_steps.append(play(mode='value_iteration', gamma=1, is_slippery=True, step_reward=-0.05, hole_reward=-2.0))
    
    
    for i in total_steps:
        print(i)