import gymnasium as gym

import torch
import hydra

from core.agent import DDPG

def train(env, conf):
    model = DDPG(env=env, conf=conf)

    if conf['load_model_for_train']:
        model.actor.load_state_dict(torch.load(conf['actor_model']))
        model.critic.load_state_dict(torch.load(conf['critic_model']))
        print(f"Successfully loaded.", flush=True)
    
    model.learn()

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
        
    env = gym.make(args['env'], render_mode='rgb_array')

    train(env, conf=args)

if __name__ == '__main__':
    main()