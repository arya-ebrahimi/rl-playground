from collections import defaultdict
from tqdm import trange
import numpy as np
import math


def argmax(Q, state, action_space_n):
    max = -math.inf
    argmax = 0
    for i in range(action_space_n):
        if Q[(state, i)] > max:
            max = Q[(state, i)]
            argmax = i
    return argmax


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
                policy[state] = argmax(Q, state, env.action_space.n)

    return Q, policy


def on_policy_monte_carlo_epsilon_soft(policy, env, num_episodes, gamma=1.0, epsilon=0.1):
    
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
            action = np.random.choice(np.arange(env.action_space.n), p=policy[state]) 
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

                A_star = argmax(Q, state, env.action_space.n)
                for a in range(env.action_space.n):
                    if a == A_star: 
                        policy[state][a] = 1 - epsilon + epsilon/env.action_space.n
                    else:
                        policy[state][a] = epsilon/env.action_space.n
                        
    return Q, policy


def generate_behaviour_policy():
    b = defaultdict()
    for sum in range(1, 33):
        for dealer in range (1, 11):
            for usable in [False, True]:
                b[(sum, dealer, usable)] = np.array([0.5, 0.5])
    return b

def off_policy_evaluation(target_policy, env, num_episodes, gamma=1.0):
    Q = defaultdict(float)
    C = defaultdict(float)

    for _ in trange(num_episodes):
        b = generate_behaviour_policy()
        # generate episode using b
        episode = []
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(np.arange(env.action_space.n), p=b[state]) 
            next_state, reward, done, _ = env.step(action)
            episode = [(state, action, reward)] + episode
            state = next_state
            
        G = 0.0
        W = 1.0        

        for timestep in episode:
            state, action, reward = timestep[0], timestep[1], timestep[2]

            G = (gamma * G) + reward
            C[(state, action)] += W
            Q[(state, action)] += (W/C[(state, action)])*(G - Q[(state, action)])
            W *= target_policy[state][action]/b[state][action]
            
    return Q    
    
    