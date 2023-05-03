import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import time

# parallelize multiple environments
num_envs = 4
envs = [lambda: gym.make("BreakoutNoFrameskip-v4") for i in range(num_envs)]
env = SubprocVecEnv(envs)

init_obs = envs.reset()

for i in range(500):
    action = envs.action_space.sample()
    obs, reward, done, info, _ = envs.step(action)
    envs.render()
    time.sleep(0.01)
envs.close()