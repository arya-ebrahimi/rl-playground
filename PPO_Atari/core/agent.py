import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import MultivariateNormal, Categorical


from core.networks import ActorCritic

class PPO:
    
    def __init__(self, env:gym.Env, conf):
        self.conf = conf
        self.env = env
        self.device = torch.device('cuda')
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.discrete = True
            # self.action_space = env.single_action_space.n
            self.action_space = env.single_action_space.n
            
        else:
            self.discrete = False
            self.action_space = env.action_space.shape[0]
            
        self.obs_space = env.observation_space.shape[0]
        
        self.agent = ActorCritic(4, self.action_space).to(self.device)
        # self.actor = MLP(input_dim=self.obs_space, output_dim=self.action_space, hidden_dim=self.conf['hidden_space']).to(self.device)
        # self.critic = MLP(input_dim=self.obs_space, output_dim=1, hidden_dim=self.conf['hidden_space']).to(self.device)
        
        self.optimizer = Adam(self.agent.parameters(), lr=self.conf['lr'])
        # self.critic_optimzer = Adam(self.critic.parameters(), lr=self.conf['lr'])
        
        # self.cov_var = torch.full(size=(self.action_space, ), fill_value=0.5).to(self.device)
        # self.cov_mat = torch.diag(self.cov_var).to(self.device)
        
    def learn(self, total_timesteps):
        timestep = 0
        iter = 0
        
        while timestep < total_timesteps:
            
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            
            timestep += np.sum(batch_lens)
            iter +=1
            
            V, _, _ = self.evaluate(batch_obs, batch_acts)
            A = batch_rtgs - V.detach()
            
            A = (A - A.mean())/(A.std() + 1e-10)
            
            for _ in range(self.conf['n_updates_per_iteration']):
                
                # print(batch_obs.shape)
                # print(self.env.single_action_space)
                
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                print(curr_log_probs.shape)
                print(batch_log_probs.shape)
                
                
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                
                surr1 = ratios * A
                surr2 = torch.clamp(ratios, 1 - self.conf['clip'], 1 + self.conf['clip']) * A
                
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs.unsqueeze(1))
                entropy_loss = entropy.mean()
                
                loss = actor_loss + self.conf['critic_coef'] * critic_loss - self.conf['entropy_coef'] * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
                
                # self.critic_optimzer.zero_grad()
                # critic_loss.backward()
                # self.critic_optimzer.step()
                
            
            if iter % self.conf['save_freq'] == 0:
                torch.save(self.agent.state_dict(), './ppo.pth')
                # torch.save(self.critic.state_dict(), './ppo_critic.pth')
                print(f'iteration: {iter}, timesteps: {timestep}')
                
    
    def rollout(self):
		# Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = torch.zeros((1024, 4)).to(self.device)
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        ep_rews = []

        t = 0 

        while t < self.conf['timesteps_per_batch']:
            ep_rews = [] # rewards collected per episode

            obs, _= self.env.reset()
            done = False
            
            for ep_t in range(self.conf['max_timesteps_per_episode']):
                
                if self.conf['render'] and len(batch_lens) == 0:
                    self.env.render()

                t += 1 # Increment timesteps ran this batch so far

                batch_obs.append(obs)
                
                action, log_prob, _ = self.get_action(obs)

                # if self.discrete:
                #     action = np.argmax(action)
                # print(action)

                obs, rew, done, _, _= self.env.step(action)
                # print(rew)
                
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs[ep_t] = log_prob
                # if done:
                #     break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            
            # print(ep_rews)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float).to(self.device)
        # batch_log_probs = np.array(batch_log_probs)
        batch_obs = batch_obs.reshape((-1, 4, 84, 84))
        batch_acts = batch_acts.reshape((-1, ))

        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
        batch_log_probs = batch_log_probs.reshape((-1, ))
        batch_rtgs = self.compute_rtgs(batch_rews).to(self.device) 
        batch_rtgs = batch_rtgs.reshape((-1, ))
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens        


    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.conf['gamma']
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):
        mean, _ = self.agent(obs)
        
        # mean = torch.nn.functional.softmax(mean, dim=-1)
        dist = Categorical(mean)

        # print(obs.shape)
        action = dist.sample()
        # print(action.shape)
        log_prob = dist.log_prob(action)
        # print(log_prob.shape)
        # print(log_prob.shape)
        return action.cpu().detach().numpy(), log_prob.detach(), dist.entropy()

    def evaluate(self, batch_obs, batch_acts):
        
        mean, V = self.agent(batch_obs)

        # print(batch_obs.shape)
        # print(batch_acts.squeeze().shape)
        dist = Categorical(mean)
        log_probs = dist.log_prob(batch_acts.squeeze())
        # print(log_probs.shape)
        return V, log_probs.detach(), dist.entropy()