#!/usr/bin/env python
# coding: utf-8

# # Collaboration and Competition
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[ ]:


from unityagents import UnityEnvironment
import numpy as np


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Tennis.app"`
# - **Windows** (x86): `"path/to/Tennis_Windows_x86/Tennis.exe"`
# - **Windows** (x86_64): `"path/to/Tennis_Windows_x86_64/Tennis.exe"`
# - **Linux** (x86): `"path/to/Tennis_Linux/Tennis.x86"`
# - **Linux** (x86_64): `"path/to/Tennis_Linux/Tennis.x86_64"`
# - **Linux** (x86, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86"`
# - **Linux** (x86_64, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Tennis.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Tennis.app")
# ```

# In[ ]:


env = UnityEnvironment(file_name="Tennis.app")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[ ]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
# 
# The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
# 
# Run the code cell below to print some information about the environment.

# In[ ]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agents and receive feedback from the environment.
# 
# Once this cell is executed, you will watch the agents' performance, if they select actions at random with each time step.  A window should pop up that allows you to observe the agents.
# 
# Of course, as part of the project, you'll have to change the code so that the agents are able to use their experiences to gradually choose better actions when interacting with the environment!

# In[ ]:


for i in range(1, 2):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))


# When finished, you can close the environment.


#***--------------BEGIN MY STUFF


from ddpg import DDPGagent
from utils import *

load_modelz = False

modelz_list = []

agent = DDPGagent(load_modelz, modelz_list, env_info)
noise = OUNoise(env_info.previous_vector_actions)
batch_size = 20
rewards = []
avg_rewards = []

#env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
#states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
all_scores = []
last_20 = []
max_games = 0
noise_set = True    #**do we want temporary exploration?
total20 = 0
train_model = True      #**Do we wish to train model? or just play the game?
LR_update_max = 10
thirty_in_row = 0
stop_training = False
throttle_model_update = 0
throttle_model_max = 10 #**well just update the model every 10 steps.
alternate_noise = True
twenty_agents = True
train_side = 0
env_train = False
alterate_batch_mem_rackets = True

test_throttle = 0

#------------
print("Begin Training")
while max_games != 5000 and stop_training == False:
    if max_games == 4000:
        env_train = False
    env_info = env.reset(train_mode=env_train)[brain_name]  # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)  #**2 X 24 (TWO AGENTS!)
    max_games += 1
    if max_games % 20 == 0: #**We have plenty of hard drive so well save every 20 games!
        #agent.save_models()
        if len(last_20) > 5:
            total20 = 0
            for idx in last_20:
                total20 += idx
            avg = total20 / len(last_20)
            print("Moving Avg: " + str(avg))
            last_20 = [] #**Reset previous 20 scores!
    print("Game Number: " + str(max_games))
    t_step = 0

    #-----------------------START TRAINING BRO
    while True:
        actions_list = []
        noisey_actions = []
        LEFT_OR_RIGHT = 0
        for individual_state in states:
            individual_state = individual_state.reshape(24)
            actions = agent.get_action(individual_state, train_model, LEFT_OR_RIGHT) #*well flip to sigmod in model to get between -1 and 1
            actions = actions.reshape(2)
            actions_list.append(actions)
            LEFT_OR_RIGHT += 1
        actions_list = np.asarray(actions_list)

        if noise_set:
            for individual_action in actions_list:
                actions = noise.get_action(individual_action, t_step)
                noisey_actions.append(actions)
            noisey_actions = np.asarray(noisey_actions)

        if max_games == 500 and noise_set and alternate_noise:  #**Use exploration just to get a baseline
            noise_set = False
            print("Turning off Exploration")

        if noise_set:
            env_info = env.step(noisey_actions)[brain_name]           # send all actions to tne environment
        else:
            env_info = env.step(actions_list)[brain_name]  # send all actions to tne environment

        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)

        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)

        if train_model:     #**We might not want to train!
            for idx in range(0, 2):
                #print(idx)
                states1 = states[idx]
                if noise_set:
                    actions1 = noisey_actions[idx]
                else:
                    actions1 = actions_list[idx]
                da_reward1 = rewards[idx]
                next_states1 = next_states[idx]
                dones1 = dones[idx]
                agent.memory.push(states1, actions1, da_reward1, next_states1, dones1)

        if train_model:
            #throttle_model_update = throttle_model_update + 1 #**lets update the model less often
            #if throttle_model_update >= throttle_model_max:
            shat = len(agent.memory)
            if len(agent.memory) > batch_size:
                if train_side:
                    agent.update_left(batch_size, train_model)
                else:
                    agent.update_right(batch_size, train_model)
                if train_side:
                    train_side = 0
                else:
                    train_side = 1

        states = next_states                               # roll over states to next time step
        t_step += 1     #**Time Step!
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
    float_score = np.mean(scores)
    float_score = float_score.item()
    all_scores.append(float_score)
    last_20.append(float_score) #***Calculate moving average!

    if float_score >= 30 and LR_update_max:  # **Once we know a lot, lets reduce our learning rate!
        LR_update_max = LR_update_max - 1  # **well only update the learning rate a few times.
        lr1, lr2 = agent.get_learning_rate()
        lr1 = lr1 * .10
        lr2 = lr2 * .10
        agent.update_learning_rate(lr1, lr2)

    if float_score >= 30:  # **Once we know enough, no reason to keep training! :)
        print("Scored over 30, great job little agent!")
        thirty_in_row += 1
    else:
        thirty_in_row = 0
    if thirty_in_row == 10:
        print("Appears we understand environment Stop Training little agent!")
        stop_training = True



#***------------END MY STUFF
# In[ ]:


env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
