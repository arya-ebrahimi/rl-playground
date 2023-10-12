import gymnasium as gym

import torch
import hydra

from core.agent import PPO
from core.utils import AtariWrapper
from gymnasium.wrappers import record_video

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def make_env(env_id, seed, idx, capture_video, run_name):
    env = gym.make(env_id, render_mode='human')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
        if idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def test(env, conf):
   model = PPO(env=env, conf=conf)

   if conf['model'] != None:
      model.agent.load_state_dict(torch.load(conf['model']))
      print(f"Successfully loaded.", flush=True)
   
   model.rollout() 
   

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
        
#    env = gym.make(args['env'], render_mode='human')
   
#    envs = gym.vector.SyncVectorEnv(
#       [make_env(args['env'], 1 + i, i, False, 'env'+str(i)) for i in range(args['num_envs'])]
#    )
   # env = record_video.RecordVideo(env, video_folder='runs', name_prefix='lunar')
   env = make_env(args['env'], 1, 1, False, 'env1')
   test(env, conf=args)
   
   env.close()

if __name__ == '__main__':
   main()