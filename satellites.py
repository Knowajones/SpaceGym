import numpy as np 
import matplotlib.pyplot as plt
import random
from time import sleep
from gym import Env, spaces

class SatellitesEnv(Env):
    def __init__(self, dmax=5, vmax=1, dt=20, fuel=50, start_min=.5, start_max=1):
        super(SatellitesEnv, self).__init__()
              
        # Define the environment hyperparameters
        self.dmax = dmax
        self.vmax = vmax
        self.burn_length = dt
        self.max_fuel = fuel
        self.start_min = start_min
        self.start_max = start_max

        # Define an observation space for the cw pos and vel vectors
        self.observation_shape = (6,)
        self.observation_space_maxs = np.array([self.dmax, self.dmax, self.dmax, self.vmax, self.vmax, self.vmax])
        self.observation_space_mins = - self.observation_space_maxs
        self.observation_space = spaces.Box(low = self.observation_space_mins, 
                                            high = self.observation_space_maxs,
                                            dtype = np.float64)
    
        # Define an action space ranging for discrete burns along each axis
        self.action_space = spaces.Discrete(7,)
        
    def reset(self):
        # Reset the fuel consumed
        self.fuel_left = self.max_fuel

        # Determine a place to intialize the chaser in
        rbar0 = random.randint(self.start_min*10, self.start_max*10)/10 * (1 if random.random() < 0.5 else -1)
        vbar0 = random.randint(self.start_min*10, self.start_max*10)/10 * (1 if random.random() < 0.5 else -1)
        zbar0 = random.randint(self.start_min*10, self.start_max*10)/10 * (1 if random.random() < 0.5 else -1)
        
        # Intialise the chaser
        self.chaser = Chaser()
        self.chaser.set_state(rbar0, vbar0, zbar0)

        # return the observation
        return self.chaser.state
    
    def render(self, mode = "human", animate=True): 
        assert mode in ["human", "state_vec"], "Invalid mode, must be either \"human\" or \"state_vec\""
        if mode == "human":
            # Define the render plot
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d',xlim = (-self.dmax,self.dmax), ylim = (-self.dmax,self.dmax),zlim = (-self.dmax,self.dmax))
            ax.scatter(0,0,0, marker='.', color='b', label='target')
            ax.set_xlabel('r-direction (km)')
            ax.set_ylabel('v-direction (km)')
            ax.set_zlabel('z (km)')
            ax.set_title('CW Frame')
            ax.scatter(self.chaser.past_states[0,:], self.chaser.past_states[1,:], self.chaser.past_states[2,:], marker='.', color='g', alpha=0.5, s=1)
            ax.scatter(self.chaser.state[0], self.chaser.state[1], self.chaser.state[2], marker='.', color='r', label='chaser')
            ax.legend()
             
            if animate:
                ax.quiver(self.chaser.state[0], self.chaser.state[1], self.chaser.state[2],self.delta_v[0]*100,self.delta_v[1]*100,self.delta_v[2]*100,color = 'red', alpha = .8)
                plt.draw()
                plt.pause(.1)
            else:
                plt.show()

        elif mode == "state_vec":
            return self.chaser.state

    def get_action_meanings(self):
        return {0: "- rbar burn", 1: "+ rbar burn", 2: "- vbar burn", 3: "+ vbar burn", 4: "- zbar burn", 5: "+ zbar burn", 6: "no burn"}

    def beyond_observation(self):
        if np.any(np.abs(self.chaser.state[:3]) >= self.dmax) or np.any(np.abs(self.chaser.state[3:]) >= self.vmax):
            return True
        return False
    
    def reward_function(self, action, done):
        reward = 0 # initialize the reward for this step
        goal_dist = .1 # reward the chaser achieving this distance or closer from the target 
        dist = np.linalg.norm(self.chaser.state[:3]) # solve for the chaser's distance
        
        # punishment for using fuel
        #reward -= 0 if action == 6 else 1
        #reward -= self.fuel_left * 0.01 if done else 0

        # reward for being close to target
        #reward += 5 if dist < goal_dist else 0
        #reward += self.dmax - dist
        reward -= dist

        # punishment for entering collision zone
        #reward -= 10 if dist < goal_dist else 0

        return int(reward)

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        # if there was a burn, decrease the fuel and penalize the reward
        if action < 6:
            self.fuel_left -= 1 
            # pass
        
        # apply the action to the chaser
        if action == 0:
            self.delta_v = np.array([-.01,0,0])
        elif action == 1:
            self.delta_v = np.array([0.01,0,0])
        elif action == 2:
            self.delta_v = np.array([0,-.01,0])
        elif action == 3:
            self.delta_v = np.array([0,0.01,0])
        elif action == 4:
            self.delta_v = np.array([0,0,-.01])
        elif action == 5:
            self.delta_v = np.array([0,0,0.01])
        elif action == 6:
            self.delta_v = np.array([0,0,0])
        
        self.chaser.past_states = np.column_stack((self.chaser.past_states, self.chaser.state))
        self.chaser.burn(self.delta_v, self.burn_length)

        # If out of fuel, end the episode.
        if self.fuel_left == 0:
            done = True

        # If outside max distance or velocity, end the episode
        if self.beyond_observation():
            done = True

        # recieve a reward based on distance to target and fuel used
        reward = self.reward_function(action, done)

        return self.chaser.state, reward, done, {'episode': None, 'is_success': None, 'fuel_left': self.fuel_left}

class Chaser(object):
    def __init__(self):
        self.state = np.zeros((6,))
        self.past_states = np.zeros((6,1))
        self.R = 405 + 6371 # km
        self.mu = 398600.5  # km^3/s^2
        self.w = np.sqrt(self.mu/self.R**3) # angular velocity

    def set_state(self, rbar_loc, vbar_loc, zbar_loc, rbar_vel=0, vbar_vel=0, zbar_vel=0):
        self.state = np.array([rbar_loc, vbar_loc, zbar_loc, rbar_vel, vbar_vel, zbar_vel])
    
    def burn(self, dv, dt):
        loc = self.state[:3].reshape((3,1))
        vel = (self.state[3:] + dv).reshape((3,1))
        wt = self.w * dt
        swt = np.sin(wt)
        cwt = np.cos(wt)
        RR = np.array([[4 - 3*cwt, 0, 0],
                       [6*(swt - wt), 1, 0],
                       [0, 0, cwt]])
        RV = np.array([[swt, 2*(1 - cwt), 0],
                       [2*(cwt - 1), 4*swt - 3*wt, 0],
                       [0, 0, swt]]) * (1/self.w)
        VR = np.array([[3*self.w*swt, 0, 0],
                       [6*self.w*(cwt - 1), 0, 0],
                       [0, 0, -self.w*swt]])
        VV = np.array([[cwt, 2*swt, 0],
                       [-2*swt, 4*cwt - 3, 0],
                       [0, 0, cwt]])
        rt = RR @ loc + RV @ vel
        vt = VR @ loc + VV @ vel
        self.state = np.concatenate((rt.reshape((3,)),vt.reshape((3,))))
    