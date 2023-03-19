import gym
from algorithms import *
import numpy as np

        
def frozen_lake():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    
    V, pi = policy_iteration(env=env, gamma=0.9, theta=1e-2)
    
    done = False
    state = env.reset()
    env.render()
    
    while not done:
        action = np.argmax(pi[state])
        next_state, reward, done, info = env.step(action)
        state = next_state
        env.render()
        
def taxi():
    env = gym.make('Taxi-v3')
    
    V, pi = policy_iteration(env=env, gamma=0.9, theta=1e-2)
    
    done = False
    state = env.reset()
    env.render()
    total_reward = 0
    
    while not done:
        action = np.argmax(pi[state])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        env.render()
    
    print(total_reward)

def main():
    frozen_lake()
    taxi()
    


if __name__ == "__main__":
    main()

