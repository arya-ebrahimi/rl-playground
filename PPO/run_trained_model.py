import gymnasium as gym

import torch
import hydra

from core.agent import PPO

def test(env, conf):
   model = PPO(env=env, conf=conf)

   if conf['actor_model'] != None and conf['critic_model'] != None:
      model.actor.load_state_dict(torch.load(conf['actor_model']))
      model.critic.load_state_dict(torch.load(conf['critic_model']))
      print(f"Successfully loaded.", flush=True)
   
   model.rollout() 
   

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
        
   env = gym.make(args['env'], render_mode='human')

   test(env, conf=args)

if __name__ == '__main__':
   main()