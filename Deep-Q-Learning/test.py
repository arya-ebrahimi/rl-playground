from dqn import QAgent
import gymnasium as gym

env = gym.make('Taxi-v3')

q = QAgent(env=env)
q.train()