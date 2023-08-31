import gymnasium as gym
import numpy as np
import math

def bellman_update(P, nA, V, pi, s, gamma):
    sum = 0
    for a in range(nA):
        for prob, next_state, reward, done in P[s][a]:
            sum += pi[s, a] * prob * (reward + gamma * V[next_state])
    V[s] = sum

def policy_evaluation(P, nS, nA, pi, V, gamma=1.0, theta=1e-5):
    
    delta = math.inf
    
    while delta > theta:
        delta = 0
        
        for s in range(nS):
            v = V[s]
            bellman_update(P, nA, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))
    return V

def q_greedify_policy(P, nA, V, pi, s, gamma):
    x = []
    for a in range(nA):
        sum = 0
        for prob, next_state, reward, done in P[s][a]:
            sum += prob * (reward + gamma * V[next_state])
        x.append(sum)

    pi[s, :] = 0
    pi[s, np.argmax(x)] = 1

def policy_improvement(P, nS, nA, V, pi, gamma=1.0):
    policy_stable = True
    
    for s in range(nS):
        old = pi[s].copy()
        q_greedify_policy(P, nA, V, pi, s, gamma)
        
        if not np.array_equal(pi[s], old):
            policy_stable = False
            
    return pi, policy_stable


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-4):
    '''
    parameters:
        P: transition probability matrix
        nS: number of states
        nA: number of actions
        gamma: discount factor
        tol: tolerance for convergence
    returns:
        value_function: value function for each state
        policy: policy for each state
    '''
    # initialize value function and policy
    value_function = np.zeros(nS)
    policy = np.ones((nS, nA)) / nA

    policy_stable = False
    
    while not policy_stable:
        value_function = policy_evaluation(P, nS, nA, pi=policy, V=value_function, gamma=gamma, theta=tol)
        policy, policy_stable = policy_improvement(P, nS, nA, V=value_function, pi=policy, gamma=gamma)
    
    return value_function, policy

if __name__ == "__main__":
    
    # create FrozenLake environment note that we are using a deterministic environment change is_slippery to True to use a stochastic environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    # reset environment to start state
    env.reset()
    # run policy iteration algorithm
    value_function, policy = policy_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=0.9, tol=1e-4)
    
    
    