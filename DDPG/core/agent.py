import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import MultivariateNormal, Categorical

from core.utils import ReplayMemory, Transition, Noise
from core.networks import MLP

from tqdm import trange

class DDPG:
    
    def __init__(self, env:gym.Env, conf):
        self.conf = conf
        self.env = env
        self.device = torch.device('cuda')
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.discrete = True
            self.action_space = env.action_space.n
        else:
            self.discrete = False
            self.action_space = env.action_space.shape[0]
        
        self.obs_space = env.observation_space.shape[0]
        
        self.actor = MLP(dims=[self.obs_space, 512, 128, self.action_space], mode='actor').to(self.device)
        self.critic = MLP(dims=[self.obs_space, 1024, 1024+self.action_space, 512, 256, 1], mode='critic').to(self.device)
                
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.conf['actor_lr'])
        self.critic_optimzer = Adam(self.critic.parameters(), lr=self.conf['critic_lr'])
        
        self.actor_target = MLP(dims=[self.obs_space, 512, 128, self.action_space], mode='actor').to(self.device)
        self.critic_target = MLP(dims=[self.obs_space, 1024, 1024+self.action_space, 512, 256, 1], mode='critic').to(self.device)
        

        self.replay_buffer = ReplayMemory(self.conf['buffer_size'])
        
        self.noise = Noise(self.action_space)
        
    def learn(self):
        
        for episode in trange(self.conf['number_of_episodes']):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.get_action(obs, noise=True)   
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                
                self.replay_buffer.push(obs, action, next_obs, reward, done)
                
                self.optimize()
                
                if done:
                    break
                
                obs = next_obs
                
            if episode % self.conf['save_freq'] == 0:
                torch.save(self.actor.state_dict(), './ddpg_actor.pth')
                torch.save(self.critic.state_dict(), './ddpg_critic.pth')
                
    def optimize(self):
        
        if len(self.replay_buffer) < self.conf['batch_size']:
            return
        
        
        transitions = self.replay_buffer.sample(self.conf['batch_size'])
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        
        # update critic
                
        next_actions = self.actor_target(next_state_batch)
        bootstraped_value = reward_batch.unsqueeze(1) + (self.conf['gamma'] * self.critic_target(next_state_batch, next_actions) * (1-done_batch).unsqueeze(1))
        expected_value = self.critic(state_batch, action_batch)
        
        
        critic_loss = nn.MSELoss()(bootstraped_value, expected_value)
        self.critic_optimzer.zero_grad()
        critic_loss.backward()
        self.critic_optimzer.step()
        
        # print(critic_loss)
        
        # update actor
        
        actions_pred = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # print(actor_loss)
        
        # update target nets
        
        for target, local in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(self.conf['tau'] * local.data + (1.0 - self.conf['tau']) * target.data)
        
        for target, local in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(self.conf['tau'] * local.data + (1.0 - self.conf['tau']) * target.data)
                        
    

    def get_action(self, obs, noise=False):
        action = self.actor(obs).cpu().detach().numpy()
        
        if noise:
            action += self.noise.noise()

        return np.clip(action, -1.0, 1.0)
    
    def rollout_actor(self):

        obs, _ = self.env.reset()
        done = False
        
        while not done:
            self.env.render()
            action = self.get_action(obs, noise=False)   
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            if done:
                break
            
            obs = next_obs