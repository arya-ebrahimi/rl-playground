import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np
from gymnasium.wrappers import record_video
import numpy as np
from collections import defaultdict
from core.algorithms import *


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
        self.action_space = Discrete(10)
        
        
    def step(self, a):
        if a < 6:
            observation, reward, terminated, truncated, info = self.env.step(a)
        else: 
            if a == 6: # south east
                action_1, action_2 = 0, 2
            elif a == 7: # sout west
                action_1, action_2 = 0, 3
            elif a == 8:
                action_1, action_2 = 1, 2
            elif a == 9:
                action_1, action_2 = 1, 3

            current_state = self.env.s
            transition1 = self.env.P[current_state][action_1]
            transition2 = self.env.P[current_state][action_2]
            can_go = False
            which_way = 0

            s_1 = transition1[0][1]
            s_2 = transition2[0][1]
            if s_1 != current_state:
                s_z = self.env.P[s_1][action_2]
                if s_z[0][1] != s_1:
                    can_go = True
                    which_way = (action_1, action_2)
            
            elif s_2 != current_state:
                s_z = self.env.P[s_2][action_1]
                if s_z[0][1] != s_2:
                    can_go = True
                    which_way = (action_2, action_1)
            
            if can_go:
                observation, reward, terminated, truncated, info = self.env.step(which_way[0])
                observation, reward, terminated, truncated, info = self.env.step(which_way[1])
            else:
                observation, reward, terminated, truncated, info = self.env.s, -1, False, False, {}

        return observation, reward, terminated, truncated, info
    
