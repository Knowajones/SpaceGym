import gym
from gym import spaces
import numpy as np

class SatelliteEnv(gym.Env):
    def __init__(self):
        # Define the action space
        self.action_space = spaces.Discrete(3)
        
        # Define the observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        # Set the initial state of the environment
        self.reset()
        
    def reset(self):
        # Set the initial state of the environment
        self.time = 0
        self.mu = 3.986004418e14  # Earth's gravitational parameter
        self.r = 6778000  # Earth's radius
        self.a = 6778000 + 500000  # semi-major axis of the orbit
        self.e = 0  # eccentricity of the orbit
        self.i = 0  # inclination of the orbit
        self.raan = 0  # right ascension of the ascending node
        self.argp = 0  # argument of periapsis
        self.n = np.sqrt(self.mu/self.a**3)  # mean motion of the orbit
        self.M = self.n*self.time  # mean anomaly of the orbit
        self.E = self.M  # eccentric anomaly of the orbit
        self.x = self.a*(np.cos(self.E) - self.e)  # position of the satellite along x-axis
        self.y = self.a*np.sqrt(1 - self.e**2)*np.sin(self.E)  # position of the satellite along y-axis
        self.vx = -self.a*self.n*np.sin(self.E)  # velocity of the satellite along x-axis
        self.vy = self.a*self.n*np.sqrt(1 - self.e**2)*np.cos(self.E)  # velocity of the satellite along y-axis
        
        # Set the initial state of the agent's satellite
        self.agent_x = self.x + 1000  # position of the agent's satellite along x-axis
        self.agent_y = self.y  # position of the agent's satellite along y-axis
        self.agent_vx = self.vx  # velocity of the agent's satellite along x-axis
        self.agent_vy = self.vy + 1000*self.n  # velocity of the agent's satellite along y-axis
        
        return np.array([self.agent_x - self.x, self.agent_y - self.y, self.agent_vx - self.vx, self.agent_vy - self.vy, self.x, self.y], dtype=np.float32)
        
    def step(self, action):
        # Perform the action
        if action == 1:  # burn in the direction of tangential motion
            self.agent_vx += 10
        elif action == 2:  # burn against the direction of tangential motion
            self.agent_vx -= 10
            
        # Update the state of the environment
        self.time += 1
        self.M = self.n*self.time
        self.E = self.M
        self.x = self.a*(np.cos(self.E) - self.e)
        self.y = self.a*np.sqrt(1 - self.e**2)*np.sin(self.E)
        self.vx = -self.a*self.n*np.sin(self.E)
        self.vy = self.a*self.n*np.sqrt(1 - self.e**2)*np.cos(self.E)
        self.agent_x += self.agent_vx
        return np.array([self.x, self.y, self.vx, self.vy])

def randomplay():
    env = SatelliteEnv()
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

randomplay()