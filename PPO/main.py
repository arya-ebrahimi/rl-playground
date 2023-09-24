import gymnasium as gym

import torch
import hydra

from core.agent import PPO

def train(env, conf):
    model = PPO(env=env, conf=conf)

    model.learn(total_timesteps=20000000)

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
    
    env = gym.make('Pendulum-v1')
    train(env, conf=args)

if __name__ == '__main__':
    main()