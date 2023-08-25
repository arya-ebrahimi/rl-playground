from collections import defaultdict
from tqdm import trange
import numpy as np
import math
import gymnasium as gym
import random

def q_learning(env:gym.Env, num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []
    for _ in trange(num_episodes):
        cum_reward = 0
        state, _ = env.reset()
        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            cum_reward+=reward
            
            Q_s = q_table[state, action]
            max_Q_s_prime = np.max(q_table[next_state])

            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * max_Q_s_prime - Q_s)
            state = next_state
        rewards.append(cum_reward)
    
    return q_table, rewards
        
def sarsa(env:gym.Env, num_episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    rewards = []
    
    for _ in trange(num_episodes):
        cum_reward = 0
        state, _ = env.reset()
        done = False
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        while not done:
            
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            cum_reward+=reward
            
            if random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[next_state])
            
            Q_s = q_table[state, action]
            Q_s_prime = q_table[next_state, next_action]

            q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * Q_s_prime - Q_s)

            state = next_state
            action = next_action
        rewards.append(cum_reward)
    
    return q_table, rewards


def argmax(Q, state, action_space_n):
    max = -math.inf
    argmax = 0
    for i in range(action_space_n):
        if Q[(state, i)] > max:
            max = Q[(state, i)]
            argmax = i
    return argmax


def on_policy_monte_carlo_epsilon_soft(policy, env, num_episodes, gamma=1.0, epsilon=0.1):
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(float)
    
    rewards = []
    
    for _ in trange(num_episodes):
        
        cum_reward = 0
        # generating episode 
        # episode list is calculated in reverse order for the simplicity of the next for loop
        episode = []
        state, _ = env.reset()
        done = False
        while not done:
            action = np.random.choice(np.arange(env.action_space.n), p=policy[state]) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            cum_reward += reward
            done = terminated or truncated
            episode = [(state, action, reward)] + episode
            state = next_state

        rewards.append(cum_reward)
        
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

                A_star = argmax(Q, state, env.action_space.n)
                for a in range(env.action_space.n):
                    if a == A_star: 
                        policy[state][a] = 1 - epsilon + epsilon/env.action_space.n
                    else:
                        policy[state][a] = epsilon/env.action_space.n
                        
    return Q, rewards
