from IPython import display
from satellites import SatellitesEnv
import msvcrt

def random_play():
    env = SatellitesEnv(fuel=50)
    obs = env.reset()
    done = False
    tot_reward = 0

    while not done:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        tot_reward += reward
        # Render the game
        env.render()

    print(tot_reward)

def human_play():
    env = SatellitesEnv(fuel=50)
    obs = env.reset()
    done = False
    tot_reward = 0
    step = 0

    while not done:
        # Take a random action
        step += 1
        if msvcrt.kbhit():
            action = int(msvcrt.getch())
            tot_reward -= 1
        else:
            action = 6
        print("Step {}: Action = {}".format(step, action))
        obs, reward, done, _ = env.step(action)
        tot_reward += reward
        # Render the game
        env.render()
    #env.render(animate=False)
    print(tot_reward)
    

def noaction_play():
    env = SatellitesEnv()
    obs = env.reset()
    done = False
    tot_reward = 0
    step = 0

    while not done:
        # Take a random action
        action = 2 if step < 0 else 6
        step += 1
        
        obs, reward, done, _ = env.step(action)
        tot_reward += reward
        # Render the game
    env.render(animate=False)

    print(tot_reward)

#noaction_play()
#random_play()
human_play()