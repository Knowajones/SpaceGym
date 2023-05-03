from IPython import display
from chopper import ChopperScape
import msvcrt

def randomplay():
    env = ChopperScape()
    obs = env.reset()

    while True:
        # Take a random action
        action = env.action_space.sample()
        print(action)
        obs, reward, done, info = env.step(action)
        
        # Render the game
        env.render()
        
        if done == True:
            break

    env.close()

def humanplay():
    env = ChopperScape()
    env.reset()
    done = False
    
    while not done:
        if msvcrt.kbhit():
            action = int(msvcrt.getch())
        else:
            action = 0
        print(action)
        obs, reward, done, info = env.step(action)
        
        env.render()
    
    env.close()

randomplay()