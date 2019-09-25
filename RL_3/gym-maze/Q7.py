import sys
import numpy as np
import gym
import gym_maze
import random
import math
import matplotlib.pyplot as plt
from collections import Counter
import time

def sarsa(learning_rate,discount_factor,explore_rate,episode,maximum_steps):
    '''Function for SARSA 
    Parameters are as :
    learning_rate : learning rate
    discount_factor : discount factor
    explore_rate : exploration rate
    episode : number of episode
    maximum_steps : maximum number of steps
    '''
    # State initial 
    state_initial=0
    # Q- table initialsed
    q_table=[]
    for i in range(25):
        s=[]
        for j in range(4):
            s.append(0)
        q_table.append(s)
    # print(q_table)
    total_reward1=-math.inf
    total_reward=0
    counter=0
    global iterations
    global steps
    iterations=[]
    steps=[]
    count=0
    for i in range(episode):
        iterations.append(i)
        steps.append(count)
        env.reset()
        if total_reward==total_reward1:
                counter=counter + 1
        if counter==5:
            print("Episodes taken for Convergence : ",i-5)
            break
        total_reward1=total_reward
        total_reward=0
        
        count=0
        while True:
            count=count+1
            env.render()
        
            p=0
            p=random.random()
            if p<explore_rate:
                action=env.action_space.sample()
            else:
                count2=0
                l=q_table[state_initial][0]
                for k in range(len(q_table[state_initial])):
                    if l ==q_table[state_initial][k]:
                        count2=count2+1
                if count2==len(q_table[state_initial]):
                    action=random.randint(0,3)
                   
                else:
                    action=q_table[state_initial].index(max(q_table[state_initial]))
                pass
            
            obs,reward,done,extras = env.step(action)
            total_reward=total_reward + reward
            
            
            p=0
            p=random.random()
            if p<explore_rate:
                action1=env.action_space.sample()
            else:
                count2=0
                l=q_table[state_initial][0]
                for k in range(len(q_table[state_initial])):
                    if l ==q_table[state_initial][k]:
                        count2=count2+1
                if count2==len(q_table[state_initial]):
                    action1=random.randint(0,3)
                   
                else:
                    action1=q_table[obs[0]*5+obs[1]].index(max(q_table[obs[0]*5+obs[1]]))
                pass
            
            q_alpha=q_table[obs[0]*5+obs[1]][action1]
            q_table[state_initial][action]=q_table[state_initial][action]+learning_rate*(reward + discount_factor*q_alpha - q_table[state_initial][action])
            state_initial=obs[0]*5+obs[1]
            if state_initial==24:
                break

def q_learning(learning_rate,discount_factor,explore_rate,episode,maximum_steps):
    '''Function for Q-learning 
    Parameters are as :
    learning_rate : learning rate
    discount_factor : discount factor
    explore_rate : exploration rate
    episode : number of episode
    maximum_steps : maximum number of steps
    '''
    #Maximum Steps
    state_initial=0
    q_table=[]
    for i in range(25):
        s=[]
        for j in range(4):
            s.append(0)
        q_table.append(s)
    # print(q_table)
    total_reward1=-math.inf
    total_reward=0
    counter=0
    global iterations
    global steps
    iterations=[]
    steps=[]
    count=0
    for i in range(episode):
        iterations.append(i)
        steps.append(count)
        env.reset()
        if total_reward==total_reward1:
                counter=counter + 1
        if counter==5:
            print("Episodes taken for Convergence : ",i)
            break
        total_reward1=total_reward
        total_reward=0
        count=0
        while True:
            count=count+1
            env.render()
            p=random.random()
            if p<explore_rate:
                action=env.action_space.sample()
            else:
                count2=0
                l=q_table[state_initial][0]
                for k in range(len(q_table[state_initial])):
                    if l ==q_table[state_initial][k]:
                        count2=count2+1
                if count2==len(q_table[state_initial]):
                    action=random.randint(0,3)
                   
                else:
                    action=q_table[state_initial].index(max(q_table[state_initial]))
                pass
            obs,reward,done,extras = env.step(action)
            total_reward=total_reward + reward
            q_alpha=max(q_table[obs[0]*5+obs[1]])
            q_table[state_initial][action]=q_table[state_initial][action]+learning_rate*(reward+discount_factor*q_alpha-q_table[state_initial][action])
            state_initial=obs[0]*5+obs[1]
            if state_initial==24 or done ==True:
                
                break
 
steps=[]
iterations=[]
if __name__=='__main__':
    count1=0
    env = gym.make("maze-random-5x5-v0")
    print("1. Q - learning ")
    print("2. SARSA ")
    # learning rate, discount factor and exploration rate
    l=float(input("Enter learning rate : "))
    d=float(input("Enter discount factor : "))
    e = float(input("Enter exploration rate : "))
    episode = int(input("Enter number of episodes : "))
    max_step = int(input("Enter maximum steps : "))
    while True:
        # When convergence happened with the optimal policy
        start_time = time.time()
        choice=int(input("Enter the Choice : "))
        if choice ==1:
            q_learning(l,d,e,episode,max_step)
            plt.plot(iterations[1:],steps[1:])
            plt.ylabel('Steps')
            plt.xlabel('Episodes')
            plt.title("Q- Learning Analysis")
            plt.show()
        elif choice ==2:
            sarsa(l,d,e,episode,max_step)
            plt.plot(iterations[1:],steps[1:])
            plt.ylabel('Steps')
            plt.xlabel('Episodes')
            plt.title("SARSA Learning  Analysis")
            plt.show()
        else:
            exit()
        print("Time :" , (time.time() - start_time))
        count1=count1+1
    
     
    
    
    
  
    

    
     
    
    
    
  
    
