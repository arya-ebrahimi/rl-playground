from collections import defaultdict
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import math

def first_visit_mc_prediction_state_value(policy, env, num_episodes, gamma=1.0):
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    
    for _ in trange(num_episodes):
        # generating episode 
        # episode list is calculated in reverse order for the simplicity of the next for loop
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode = [(state, action, reward)] + episode
            state = next_state

        G = 0
        first_visit = set()
    
        # calculating state value for the first visited state
        for timestep in episode:
            state, action, reward = timestep[0], timestep[1], timestep[2]
            if state not in first_visit:
                first_visit.add(state)
                G = (gamma * G) + reward
                returns_sum[state] += G
                returns_count[state] += 1.0
                
                V[state] = returns_sum[state] / returns_count[state]

    return V   

def monte_carlo_exploring_starts(policy, env, num_episodes, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(float)
    
    for _ in trange(num_episodes):
        # generating episode 
        # episode list is calculated in reverse order for the simplicity of the next for loop
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done, _ = env.step(action)
            episode = [(state, action, reward)] + episode
            state = next_state

        G = 0
        first_visit = set()
    
        # calculating state value for the first visited state
        for timestep in episode:
            state, action, reward = timestep[0], timestep[1], timestep[2]
            if (state, action) not in first_visit:
                first_visit.add((state, action))
                G = (gamma * G) + reward
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1.0
                
                Q[(state, action)] = returns_sum[(state, action)] / returns_count[(state, action)]
                
                def argmax(Q, state):
                    max = -math.inf
                    argmax = 0
                    for i in range(env.action_space.n):
                        if Q[(state, i)] > max:
                            max = Q[(state, i)]
                            argmax = i
                    return argmax
                        
                policy[state] = argmax(Q, state)

    return Q, policy