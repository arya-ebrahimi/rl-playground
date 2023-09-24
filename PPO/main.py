import gymnasium as gym

import torch
import hydra


def train():
    model = 2


@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main():
    
    env = gym.make('Pendulum-v0')

if __name__ == '__main__':
    main()