from collections import defaultdict
from tqdm import trange

def monte_carlo_prediction(policy, env, num_episodes, gamma=1.0):
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    
    for _ in trange(num_episodes):

        episode = []
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode = [(state, action, reward)] + episode
            state = next_state

        G = 0
        
        for timestep in episode:
            state, action, reward = timestep[0], timestep[1], timestep[2]
            
            G = (gamma * G) + reward
            returns_sum[state] += G
            returns_count[state] += 1.0
            
            V[state] = returns_sum[state] / returns_count[state]

    return V    