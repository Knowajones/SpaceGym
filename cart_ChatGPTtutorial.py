import gym

env = gym.make('CartPole-v1', render_mode='human')

import random
import numpy as np

# Set the number of episodes and maximum number of steps per episode
n_episodes = 1000
n_steps = 500

# Set the initial value of epsilon (the exploration rate)
epsilon = 1.0
# Set the minimum value of epsilon (the minimum exploration rate)
epsilon_min = 0
# Set the rate at which epsilon will decrease over time
epsilon_decay = 0.5
# Set the learning rate (alpha) and discount factor (gamma)
alpha = 0.1
gamma = 0.6

# Initialize the Q-table with zeros
q_table = np.zeros((env.observation_space.shape[0], env.action_space.n))

# Loop through a fixed number of episodes
for i_episode in range(n_episodes):
    # Reset the environment for each episode
    observation = env.reset()
    # Initialize the total reward for the episode
    total_reward = 0
    # Loop through a fixed number of steps per episode
    for t in range(n_steps):
        # Render the environment (you can remove this line to speed up training)
        env.render()
        # Select an action using an epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Choose a random action
        else:
            action = np.argmax(q_table[observation.astype(int)])  # Choose the action with the highest Q-value
        # Take a step in the environment using the selected action
        next_observation, reward, done, info = env.step(action)
        # Update the total reward for the episode
        total_reward += reward
        # Update the Q-value for the current state-action pair using the Q-learning update rule
        q_table[observation.astype(int), action] += alpha * (reward + gamma * np.max(q_table[next_observation.astype(int)]) - q_table[observation.astype(int), action])
        # Set the current observation to the next observation
        observation = next_observation
        # If the episode is done, update the epsilon parameter and print the total reward
        if done:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print('Episode {}: Total Reward = {}'.format(i_episode + 1, total_reward))
            break

# Close the environment
env.close()