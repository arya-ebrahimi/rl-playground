import gymnasium as gym

import torch
import hydra
from core.utils import AtariWrapper
from core.agent import PPO

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def train(env, conf):
    model = PPO(env=env, conf=conf)

    if conf['model'] != None:
        model.agent.load_state_dict(torch.load(conf['model']))
        print(f"Successfully loaded.", flush=True)
   
    
    model.learn(total_timesteps=20000000)
    
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
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

    return thunk

@hydra.main(config_path="config", config_name="config.yaml", version_base=None)
def main(args):
    # ale = ALEInterface()
    # ale.loadROM(Pong)
    # env = gym.make(args['env'], render_mode='rgb_array')
    env = gym.make("ALE/Pong-v5")
    # env = AtariWrapper(env)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args['env'], 1 + i, i, False, 'env'+str(i)) for i in range(args['num_envs'])]
    )
    # a = env.reset()
    # print(a)   
    train(envs, conf=args)

if __name__ == '__main__':
    main()