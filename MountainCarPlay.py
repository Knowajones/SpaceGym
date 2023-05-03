'''
import gym
from gym.utils import play
env = gym.make('MountainCar-v0', render_mode='human')
env = play.play(env, zoom=3, keys_to_action={"a":2, "da":0}, noop=1)
'''
import gym
import msvcrt

def humanplay():
    env = gym.make("MountainCar-v0")
    env.reset()
    done = False
    tot_reward = 0
    loc = 0
    while loc <= 0.5:
        if msvcrt.kbhit():
            action = int(msvcrt.getch())
            tot_reward -= 1
        else:
            action = 1
        obs, reward, done, _ = env.step(action)
        loc = obs[0]
        print(tot_reward)
        env.render()
    print('Reward: {}'.format(tot_reward))

def hardcodeplay():
    env = gym.make("MountainCar-v0")
    obs = env.reset()
    done = False
    tot_reward = 0
    loc = obs[0]
    while loc <= 0.5:
        action = 0 if obs[1] < 0 else 2
        obs, reward, done, _ = env.step(action)
        loc = obs[0]
        tot_reward -= 1
        env.render()
    print('Reward: {}'.format(tot_reward))

hardcodeplay()