import gymnasium as gym
import cv2
import numpy as np

class AtariWrapper(gym.Wrapper):
    
    def __init__(self, env, name='pong', k=4):
        super(AtariWrapper, self).__init__(env)
        
        self.cut_height = [33, -15]
        self.cut_width = [0, -1]
        
        self.k = k
        
        self.dsize = [84, 84]
        
    def reset(self):
        self._return = 0
        self.life = 0
        
        ob, _ = self.env.reset()
        ob = self.process(ob)
        self.fstack = np.stack([ob for i in range(self.k)])
        
        return self.fstack
    
    def step(self, action):
        reward = 0
        done = False
        # print(action)
        lose_life = False
        
        frames = []
        
        for i in range(self.k):
            obs, rew, terminated, truncated, info = self.env.step(action=action)
            
            if lose_life:
                if info['ale.lives'] < self.life:
                    lose_life = True
                self.life = info['ale.lives']
            
            
            # print(obs)
            obs = self.process(obs)
            frames.append(obs)
            
            reward += rew
            
            if terminated or truncated:
                done = True
                break
            
        self.stack(frames)
        
        
        self._return += reward
        
        if done:
            info['return'] = self._return
        
        if reward > 0:
            reward = 1
        elif reward != 0:
            reward = -1
        
        # print(self.fstack)
        return self.fstack, reward, done, info, lose_life
    
    def stack(self, frames):
        n = len(frames)
        
        if n == self.k:
            self.fstack = np.stack(frames)
        elif n > self.k:
            self.fstack = np.stack(frames[-self.k::])
        else:
            self.fstack[0:self.k-n] = self.fstack[n::]
            self.fstack[self.k-n::] = np.stack(frames)
            
    def process(self, obs):
        obs = cv2.cvtColor(obs[self.cut_height[0]:self.cut_height[1],
                           self.cut_width[0]:self.cut_width[1]], cv2.COLOR_BGR2GRAY)
        obs = cv2.resize(obs, dsize=self.dsize)
    
        return obs