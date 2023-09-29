import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam

from core.utils import ReplayMemory, Transition
from core.networks import MLP, Actor, Critic

from tqdm import trange

class TD3:
    
    def __init__(self, env:gym.Env, conf):
        self.conf = conf
        self.env = env
        self.device = torch.device('cuda')
        
 
        self.discrete = False
        self.action_space = env.action_space.shape[0]
        self.obs_space = env.observation_space.shape[0]
        
        self.max_action = float(self.env.action_space.high[0])
        
        self.actor = Actor(dims=[self.obs_space, 256, 256, self.action_space], device=self.device, max_action=self.max_action).to(self.device)
        self.critic = Critic(dims=[self.obs_space + self.action_space, 256, 256, 1], device=self.device).to(self.device)
                
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.conf['actor_lr'])
        self.critic_optimzer = Adam(self.critic.parameters(), lr=self.conf['critic_lr'])
        
        self.actor_target = Actor(dims=[self.obs_space, 256, 256, self.action_space], device=self.device, max_action=self.max_action).to(self.device)
        self.critic_target = Critic(dims=[self.obs_space + self.action_space, 256, 256, 1], device=self.device).to(self.device)
        

        self.replay_buffer = ReplayMemory(self.conf['buffer_size'])
        
        
    def learn(self):
        
        for episode in trange(self.conf['number_of_episodes']):
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.get_action(obs)   
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                
                self.replay_buffer.push(obs, action, next_obs, reward, done)
                
                self.optimize(episode)
                
                if done:
                    break
                
                obs = next_obs
                
            if episode % self.conf['save_freq'] == 0:
                torch.save(self.actor.state_dict(), './td3_actor.pth')
                torch.save(self.critic.state_dict(), './td3_critic.pth')
                
    def optimize(self, episode_num):
        
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
        
        noise = (torch.randn_like(action_batch) * self.conf['policy_noise']).clamp(-self.conf['noise_clip'], self.conf['noise_clip'])
        next_actions = (self.actor_target(next_state_batch) + noise).clamp(-self.max_action, self.max_action)
        
                        
        target_q1, target_q2 = self.critic_target(next_state_batch, next_actions)
        target_q = torch.min(target_q1, target_q2)
        
        with torch.no_grad():
            bootstraped_value = reward_batch.unsqueeze(1) + (self.conf['gamma'] * target_q) * (1-done_batch).unsqueeze(1)
        
        expected_value = self.critic(state_batch, action_batch)
        
        
        critic_loss = nn.MSELoss()(bootstraped_value, expected_value[0]) + nn.MSELoss()(bootstraped_value, expected_value[1])
        self.critic_optimzer.zero_grad()
        critic_loss.backward()
        self.critic_optimzer.step()
        
        
        # update actor (delayed)
        
        if episode_num % self.conf['delay_freq'] == 0:
            actions_pred = self.actor(state_batch)
            actor_loss = -self.critic.get_q1(state_batch, actions_pred).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
            # update target nets
            
            for target, local in zip(self.actor_target.parameters(), self.actor.parameters()):
                target.data.copy_(self.conf['tau'] * local.data + (1.0 - self.conf['tau']) * target.data)
            
            for target, local in zip(self.critic_target.parameters(), self.critic.parameters()):
                target.data.copy_(self.conf['tau'] * local.data + (1.0 - self.conf['tau']) * target.data)
                        
    

    def get_action(self, obs, noise=True):
        action = self.actor(obs).cpu().detach().numpy()
        if noise:
            action += np.random.normal(0, self.max_action * self.conf['exploration_noise'], size=self.action_space)
        return np.clip(action, -self.max_action, self.max_action)
    
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