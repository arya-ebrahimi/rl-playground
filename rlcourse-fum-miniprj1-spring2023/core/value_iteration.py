import gymnasium as gym
from core.policy_iteration import q_greedify_policy
import math
import numpy as np

def bellman_optimality_update(P, nA, V, s, gamma):
    
    max_a = -math.inf
    for a in range(nA):
        for prob, next_state, reward, done in P[s][a]:
            value = prob * (reward + gamma * V[next_state])
            max_a = max(max_a, value)

    V[s] = max_a


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-4):
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
    
    delta = math.inf
    
    while delta > tol:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            bellman_optimality_update(P=P, nA=nA, s=s, V=value_function, gamma=gamma)
            delta = max(delta, abs(v - value_function[s]))

    for s in range(nS):
        q_greedify_policy(P, nA, value_function, policy, s, gamma)
    
    return value_function, policy


if __name__ == "__main__":

    # create FrozenLake environment note that we are using a deterministic environment change is_slippery to True to use a stochastic environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    # reset environment to start state
    env.reset()
    # run value iteration algorithm
    value_function, policy = value_iteration(env.P, env.observation_space.n, env.action_space.n, gamma=0.9, tol=1e-4)
