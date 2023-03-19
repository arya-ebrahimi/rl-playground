import numpy as np
from policy_iteration import q_greedify_policy
import math

def bellman_optimality_update(env, V, s, gamma):
    
    max_a = -math.inf
    for a in range(env.action_space.n):
        for prob, next_state, reward, done in env.P[s][a]:
            value = prob * (reward + gamma * V[next_state])
            max_a = max(max_a, value)

    V[s] = max_a

def value_iteration(env, gamma, theta):
    V = np.zeros(env.observation_space.n)
    
    delta = math.inf
    while delta > theta:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        
    pi = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    for s in range(env.observation_space.n):
        q_greedify_policy(env, V, pi, s, gamma)
    
    return V, pi

