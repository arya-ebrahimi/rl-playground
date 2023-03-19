import numpy as np

def bellman_update(env, V, pi, s, gamma):
    sum = 0
    for a in range(env.action_space.n):
        for prob, next_state, reward, done in env.P[s][a]:
            sum += pi[s, a] * prob * (reward + gamma * V[next_state])
    V[s] = sum

def policy_evaluation(env, pi, V, gamma=1.0, theta=1e-5):
    
    delta = float('inf')
    
    while delta > theta:
        delta = 0
        
        for s in range(env.observation_space.n):
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
    return V

def q_greedify_policy(env, V, pi, s, gamma):
    x = []
    for a in range(env.action_space.n):
        sum = 0
        for prob, next_state, reward, done in env.P[s][a]:
            sum += prob * (reward + gamma * V[next_state])
        x.append(sum)

    pi[s, :] = 0
    pi[s, np.argmax(x)] = 1

def policy_improvement(env, V, pi, gamma=1.0):
    policy_stable = True
    
    for s in range(env.observation_space.n):
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        
        if not np.array_equal(pi[s], old):
            policy_stable = False
            
    return pi, policy_stable

def policy_iteration(env, gamma=1.0, theta=1e-5):
    V = np.zeros(env.observation_space.n)
    pi = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n
    policy_stable = False
    
    while not policy_stable:
        V = policy_evaluation(env, pi, V, gamma, theta)
        pi, policy_stable = policy_improvement(env, V, pi, gamma)
        
    return V, pi