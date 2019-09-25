#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Importing the libraries needed 
# Reference : https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter06/random_walk.py
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


def initialisation_parameters():
    # 0 is the left terminal state
    # 6 is the right terminal state
    # 1 ... 5 represents A ... E
    VALUES = np.zeros(7)
    VALUES[1:6] = 0.5
    # For convenience, we assume all rewards are 0
    # and the left terminal state has value 0, the right terminal state has value 1
    # set up true state values
    TRUE_VALUE = np.zeros(7)
    TRUE_VALUE[6] = VALUES[6] = ACTION_RIGHT = 1
    TRUE_VALUE[1:6] = np.arange(1, 6) / 6.0
    ACTION_LEFT = 0
    return VALUES,TRUE_VALUE,ACTION_LEFT,ACTION_RIGHT


# In[47]:


def td_loop(state,trajectory,batch,values,alpha,rewards):
    old_state = state
    if np.random.binomial(1, 0.5) != ACTION_LEFT:
        state += 1
    else:
        state -= 1
    # Assume all rewards are 0
    reward = 0
    trajectory.append(state)
    # TD update
    if not batch:
        gt = (reward + values[state] - values[old_state])
        values[old_state] += alpha * gt
    if state == 6 or state == 0:
        return None,None,None,None,None,None
    ###
    rewards.append(reward)
    return state,trajectory,batch,values,alpha,rewards
    
def temporal_difference(values, alpha=0.1, batch=False):
    '''Function for temporal difference 
    Parameters are as : 
        values: current states value, will be updated if @batch is False
        alpha: step size
        batch: whether to update @values'''
    state = 3
    trajectory = [state]
    rewards = [0]
    while True:
        state,trajectory,batch,values,alpha,rewards = td_loop(state,trajectory,batch,values,alpha,rewards)
        if state == None:
            break
    return trajectory, rewards


# In[43]:


def conditional_monte(state):
    if np.random.binomial(1, 0.5) != ACTION_LEFT:
        state += 1
    else:
        state -= 1
    return state
    
def monte_carlo(values, alpha=0.1, batch=False):
    ''' Function for Monte Carlo
    Parameters are as:
        values: current states value, will be updated if @batch is False
        alpha: step size
        batch: whether to update values
        '''
    state = 3
    trajectory = [3]
    # if end up with left terminal state, all returns are 0
    # if end up with right terminal state, all returns are 1
    while True:
        state = conditional_monte(state)
        trajectory.append(state)
        if state == 0:
            returns = 0.0
            break
        elif state == 6:
            returns = 1
            break
    if not batch:
        for state_ in trajectory[:-1]:
            # MC update
            gt = (returns - values[state_])
            value_add = alpha * gt
            values[state_] = values[state_] + value_add
    return trajectory, [returns] * (len(trajectory) - 1)


# In[44]:


# Example 6.2 left
def compute_state_loop(episodes,current_values):
    for i in range(episodes[-1] + 1):
        if i in episodes:
            plt.plot(current_values, label=str(i) + ' episodes')
        temporal_difference(current_values)
    return current_values

def compute_state_value():
    
    episodes = [0, 1, 10, 10*2]
    
    current_values = np.copy(VALUES)
    plt.figure(1)
    
    current_values = compute_state_loop(episodes,current_values)
    plt.plot(TRUE_VALUE, label='true values')
    plt.xlabel('state')
    plt.ylabel('estimated value')
    plt.legend()


# In[45]:



# Example 6.2 right
def rms_loop(episodes,runs,total_errors,method,alpha):
    for r in tqdm(range(runs)):
            errors = []
            current_values = np.copy(VALUES)
            for i in range(0, episodes):
                power_true = np.power(TRUE_VALUE - current_values, 2)
                value_new = np.sum(power_true)
                sqrt_v = np.sqrt(value_new / 5.0)
                errors.append(sqrt_v)
                if method == 'TD':
                    temporal_difference(current_values, alpha=alpha)
                else:
                    monte_carlo(current_values, alpha=alpha)
            total_errors = total_errors + np.asarray(errors)
    return total_errors

def rms_cond(td_alphas):
    if i >= len(td_alphas):
            method = 'MC'
            linestyle = 'dashdot'
        else:
            method = 'TD'
            linestyle = 'solid'
    return method,linestyle
    
def rms_error():
    # Same alpha value can appear in both arrays
    td_alphas = [0.15, 0.1, 0.05]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    episodes = 101
    runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        total_errors = np.zeros(episodes)
        method,linestyle = rms_cond(td_alphas)
        total_errors = rms_loop(episodes,runs,total_errors,method,alpha)
        total_errors =total_errors/runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', alpha = %.02f' % (alpha))
    plt.xlabel('episodes')
    plt.ylabel('RMS')
    plt.legend()


# In[ ]:





# In[46]:


VALUES,TRUE_VALUE,ACTION_LEFT,ACTION_RIGHT = initialisation_parameters()
plt.figure(figsize=(10, 20))
###
plt.subplot(2, 1, 1)
compute_state_value()
###
plt.subplot(2, 1, 2)
rms_error()
plt.tight_layout()
plt.savefig('../images3/example_6_2.png')
plt.close()


# In[ ]:





# In[ ]:




