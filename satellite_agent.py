from stable_baselines3 import PPO
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from satellites import SatellitesEnv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SatelliteQAgent():
    def __init__(self, buckets=(20,20,20,5,5,5), num_episodes=1000,
                 min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=10):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = SatellitesEnv()

        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

        # [position, velocity]
        self.upper_bounds = list(self.env.observation_space.high)
        self.lower_bounds = list(self.env.observation_space.low)
        
        #
        self.rewards = np.zeros(self.num_episodes)

    def discretize_state(self, obs):
        """
        Takes an observation of the environment and aliases it.
        By doing this, very similar observations can be treated
        as the same and it reduces the state space so that the 
        Q-table can be smaller and more easily filled.
        
        Input:
        obs (tuple): Tuple containing 4 floats describing the current
                     state of the environment.
        
        Output:
        discretized (tuple): Tuple containing 4 non-negative integers smaller 
                             than n where n is the number in the same position
                             in the buckets list.
        """
        discretized = list()
        for i in range(len(obs)):
            scaling = ((obs[i] + abs(self.lower_bounds[i])) 
                       / (self.upper_bounds[i] - self.lower_bounds[i]))
            new_obs = int(round((self.buckets[i] - 1) * scaling))
            new_obs = min(self.buckets[i] - 1, max(0, new_obs))
            discretized.append(new_obs)
        return tuple(discretized)
    
    def choose_action(self, state):
        """
        Implementation of e-greedy algorithm. Returns an action (0 or 1).
        
        Input:
        state (tuple): Tuple containing 4 non-negative integers within
                       the range of the buckets.
        
        Output:
        (int) Returns either 0 or 1
        """
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample() 
        else:
            return np.argmax(self.Q_table[state])
        
    def update_q(self, state, action, reward, new_state):
        """
        Updates Q-table using the rule as described by Sutton and Barto in
        Reinforcement Learning.
        """
        self.Q_table[state][action] += (self.learning_rate * 
                                        (reward 
                                         + self.discount * np.max(self.Q_table[new_state]) 
                                         - self.Q_table[state][action]))

    def get_epsilon(self, t):
        """Gets value for epsilon. It declines as we advance in episodes."""
        # Ensures that there's almost at least a min_epsilon chance of randomly exploring
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        """Gets value for learning rate. It declines as we advance in episodes."""
        # Learning rate also declines as we add more episodes
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self):
        """
        Trains agent making it go through the environment and choose actions
        through an e-greedy policy and updating values for its Q-table. The 
        agent is trained by default for 500 episodes with a declining 
        learning rate and epsilon values that with the default values,
        reach the minimum after 198 episodes.
        """
        # Looping for each episode
        for e in range(self.num_episodes):
            # Initializes the state
            current_state = self.discretize_state(self.env.reset())
            
            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False
                        
            # Looping for each step
            while not done:
                
                # Choose A from S
                action = self.choose_action(current_state)
        
                # Take action
                obs, reward, done, _ = self.env.step(action)
                self.rewards[e] += reward
                new_state = self.discretize_state(obs)
                # Update Q(S,A)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state
                
                # We break out of the loop when done is False which is
                # a terminal state.
            
            print('Episode {}: Total Reward = {}, Fuel Left = {}, '.format(e + 1, self.rewards[e], self.env.fuel_left))

        print('Finished training!')
    
    def plot_learning(self):
        """
        Plots the reward at each episode and prints the
        amount of times that an episode was successfully completed.
        """
        sns.lineplot(range(len(self.rewards)),self.rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()        

    def run(self):
        """Runs an episode while displaying the environment."""
        done = False
        current_state = self.discretize_state(self.env.reset())
        self.epsilon = 1
        tot_reward = 0

        while not done:
            action = self.choose_action(current_state)
            obs, reward, done, _ = self.env.step(action)
            tot_reward += reward
            new_state = self.discretize_state(obs)
            current_state = new_state
            self.env.render()
        self.env.render(animate=False)
        
        print("Reward: {}".format(tot_reward))

    def evaluate(self, num_tests=100):
        tot_rewards = np.zeros((num_tests,))
        
        for itest in range(num_tests):
            done = False
            current_state = self.discretize_state(self.env.reset())
            self.epsilon = 1

            while not done:
                action = self.choose_action(current_state)
                obs, reward, done, _ = self.env.step(action)
                tot_rewards[itest] += reward
                new_state = self.discretize_state(obs)
                current_state = new_state

        print('Avg Reward {}: Highest Reward = {}'.format(tot_rewards.mean(), tot_rewards.max()))
        sns.scatterplot(range(num_tests),tot_rewards)
        sns.lineplot(range(num_tests),np.ones((num_tests,))*tot_rewards.mean())
        plt.xlabel("Test")
        plt.ylabel("Reward")
        plt.show()

class SatellitePPOAgent():
    def __init__(self, model_name=None):
        self.env = SatellitesEnv()
        if not model_name:
            self.model = PPO("MlpPolicy", self.env, verbose=1)
        else:
            self.model = PPO.load(model_name)

    def train(self, tot_steps):
        self.model.learn(total_timesteps=tot_steps)

    def store(self, name):
        self.model.save(name)

    def plot_learning(self):
        pass

    def evaluate(self):
        pass

    def run(self):
        done = False
        obs = self.env.reset()
        tot_reward = 0
        while not done:
            action, _ = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            tot_reward += reward
        print("Reward: {}, Fuel: {}".format(tot_reward, info.get('fuel_left')))
        self.env.render(animate=False)

def Q_main():
    
    # train the agent
    agent = SatelliteQAgent(buckets=(5,5,5,15,15,15), num_episodes=20000,
                    min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=10)
    agent.train()
    
    # save the agent
    np.save('Q_table.npy', agent.Q_table)

    # plot the learning
    agent.plot_learning()
    
    # load the agent
    agent = SatelliteQAgent()
    agent.Q_table = np.load('Q_table.npy')

    # evaluate the results of n tests
    agent.evaluate(num_tests=50)

    # show the agent in action
    agent.run()

def PPO_main():
    
    env = SatellitesEnv()
    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=1000000)
    
    #model.save("ppo_satellites")
    #del model # remove to demonstrate saving and loading
    model = PPO.load("ppo_satellites")
    #model.save("ppo_satellites3")
    
    done = False
    obs = env.reset()
    tot_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        tot_reward += reward
    print("Reward: {}, Fuel: {}".format(tot_reward, info.get('fuel_left')))
    env.render(animate=False)
    

if __name__ == "__main__":
    #Q_main()
    PPO_main()
