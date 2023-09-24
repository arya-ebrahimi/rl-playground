import gymnasium as gym

import torch
import hydra

from core.agent import PPO

def train(env, conf):
    model = PPO(env=env, conf=conf)

    if conf['actor_model'] != None and conf['critic_model'] != None:
        model.actor.load_state_dict(torch.load(conf['actor_model']))
        model.critic.load_state_dict(torch.load(conf['critic_model']))
        print(f"Successfully loaded.", flush=True)
    
    model.learn(total_timesteps=20000000)

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
        
    env = gym.make('Pendulum-v1', render_mode='human')

    train(env, conf=args)

if __name__ == '__main__':
    main()