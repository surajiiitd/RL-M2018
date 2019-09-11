#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Packages to be imported
# Referenced : https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter03/grid_world.py 
from matplotlib.table import Table
import matplotlib
import copy
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


def global_variable_declaration():
    '''Declaration of all the global variables '''
    # Global variable declaration 
    
    WORLD_SIZE = 5
    # World size
    B_POS = [0, 3]
    # B initial size
    B_PRIME_POS = [2, 3]
    # B prime position
    A_POS = [0, 1]
    # A initial position
    A_PRIME_POS = [4, 1]
    # A prime position
    DISCOUNT = 0.9
    # Discount factor
    # Equal probability will be so 0.25 since four actions are there .
    ACTION_PROB = 0.25
    # Probability of action to be taken
    # left, up, right, down
    # Set of Actions
    actions_list = [[0, -1],[-1,0],[0,1],[1,0]]
    # To convert it into numpy array
    ACTIONS = np.array(actions_list)
    # returning all the assigned global values 
    return WORLD_SIZE,A_POS,A_PRIME_POS,B_POS,B_PRIME_POS,DISCOUNT,ACTIONS,ACTION_PROB


# In[3]:


# Calling the function to store all the global variables
WORLD_SIZE,A_POS,A_PRIME_POS,B_POS,B_PRIME_POS,DISCOUNT,ACTIONS,ACTION_PROB = global_variable_declaration()


# In[4]:


def step(state, action):
    '''Function to take next step given state and actions
    Parameters are as :
    state : The present state
    action : The action to be taken 
    '''
    
    # If the state is equal to A's initial pos then return A's prime position and 10
    if state == A_POS:
        return A_PRIME_POS, 10
    # If the state is equal to B's initial pos then return B's prime position and 5
    elif state == B_POS:
        return B_PRIME_POS, 5

#     print(np.array(state).shape,action.shape)
    # Making the next state by concatenating the state and action array
    next_state = (np.array(state) + action)
#     print(next_state.shape)
    next_state=next_state.tolist()
    ###
    ###
    x= next_state[0]
    y = next_state[1]
    # Condition to assign the next state and reward
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    # Returning the next state and reward
    return next_state, reward


# In[5]:


def enumerate_table(image,width,height,tb):
    '''To enumerate the table cells
    Parameters are as :
    image : image to be plotted 
    width : width 
    height : height
    tb : tb cell table variable'''
    # Add cells

    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')
        
    return tb


# In[6]:


def add_cell(image,width,height,tb):
    '''To make the loop to add cells 
    Parameters are as:
    image: image to plot 
    width : width 
    height : height 
    tb : table variable 
    '''
    
    for i in range(len(image)):
            tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                        edgecolor='none', facecolor='none')
            tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                        edgecolor='none', facecolor='none')
    return tb


# In[ ]:





# In[7]:


def draw_table(image,ax):
    '''To draw the table 
    Parameters are as:
    image : image to make the table 
    '''
    
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = image.shape
    width = float(1.0 / ncols)
    height = float(1.0 / nrows)
    
    tb = enumerate_table(image,width,height,tb)
    # Row and column labels for table
    
    tb = add_cell(image,width,height,tb)
    ax.add_table(tb)
    


# In[8]:


def draw_image(image):
    '''To draw rhe table for given data
    Parameters are as:
    image: Image for which making the table
    '''
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    draw_table(image,ax)
    


# In[9]:


def bellmann_updation(value):
    '''Bellmannn equation updation function 
    Parameters are as:
    value : value 
    action : action taken 
    reward : reward used
    '''
    new_value = np.zeros_like(value)
    # Loop for updating the value 
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            for action in ACTIONS:
                next_v, reward = step([i, j], action)
        
                # bellman equation for updating the value
                new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_v[0], next_v[1]])
    return new_value


# In[10]:


def check_termination(value,new_value,cons):
    '''To check termination for loop
    Parameters are as:
    value : value 
    new_value : new_ value i.e, modified value
    cons :  constant to check for error
    '''
    
    absolute = np.abs(value - new_value)
    absolute_array = np.array(absolute)
#     print(np.sum(absolute_array))
    if np.sum(absolute_array) <= cons:
        ### 
        # To check the change in variable i.e, kind of error
        ###
        change = np.round(new_value, decimals=3)
        draw_image(change)
        plt.savefig('../images/figure_3_2.png')
        ## To giv ethe title
        plt.title("Table 3.2")
        plt.close()
        return True
    return False


# # Figure 3.2

# In[11]:


'''To make the table plot and save in images directory of previous location directory'''
# To make the figure
# To make the array of size WORLD_SIZE, WORLD_SIZE
value = np.zeros((WORLD_SIZE, WORLD_SIZE))

# While loop with terminating condition 
counter =0
while True:
    counter = counter +1
    # keep iteration until convergence
    # Making the duplicate of value variable 
    
    new_value = bellmann_updation(value)
    # If the the change in the value will be less than 1e-4 then terminate and save the figure
    condition = check_termination(value,new_value,cons=1e-4)
    if condition == True:
        # To  break when it is done
        break
    else:
        pass
    value = new_value


# In[ ]:




