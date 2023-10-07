import gymnasium as gym

import torch
import hydra

from core.agent import PPO
from core.utils import AtariWrapper
from gymnasium.wrappers import record_video


def test(env, conf):
   model = PPO(env=env, conf=conf)

   if conf['model'] != None:
      model.agent.load_state_dict(torch.load(conf['model']))
      print(f"Successfully loaded.", flush=True)
   
   model.rollout() 
   

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
        
   env = gym.make(args['env'], render_mode='human')
   env = AtariWrapper(env)
   # env = record_video.RecordVideo(env, video_folder='runs', name_prefix='lunar')
   test(env, conf=args)
   
   env.close()

if __name__ == '__main__':
   main()