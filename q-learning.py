# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 15:44:44 2020
This code implements Q-learning to find the optimal strategy to play a game in AI gym.

@author: Liying Lu
"""

# import libraries
import gym
import numpy as np
import random

def FrozenLake(is_slippery, exploration, learning, discount, episodes, steps):
    # create environment from AI gym
    env = gym.make('FrozenLake-v0', is_slippery=is_slippery)
    wins = 0
    env.reset()
    
    # delcare variables for q-learning
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    for i_episode in range(episodes):
        state = env.reset()
        for x in range(steps):
            
            # which step to take
            if random.uniform(0,1) < exploration:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            nextState, reward, done, info = env.step(action)
            #env.render()
            #q_table[state,action] += reward
            
            q_table[state,action] = (1-learning)*q_table[state,action]+ \
            learning*(reward + discount*q_table[nextState,np.argmax(q_table[nextState])])
            
            state = nextState
            
            if done and reward == 1:
                wins +=1
            if done:
                break
            
        exploration = exploration * (1-i_episode/episodes)
    
    #print("*************************************************************************")
    #print("is_slippery = ", is_slippery)
    #print(q_table)
    print("Learning_rate:", learning, "Number of wins: ", wins)
    env.close()
    
# test the algorithm
exploration = 1
learning = [1, .9, .8, .7, .6, .5]
discount = 0.9
episodes = 100000
steps = 150
print("is_slippery = False")
for learn_rate in learning:
    FrozenLake(False, exploration, learn_rate, discount, episodes, steps)        
print("***********************************************************************\nis_slippery = True")
for learn_rate in learning:
    FrozenLake(True, exploration, learn_rate, discount, episodes, steps)
        
        
        
        
        
        
        
        
        
        