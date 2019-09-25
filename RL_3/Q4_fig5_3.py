#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries needed 
# Reference : https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter05/blackjack.py
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def policy_player_initialisation(ACTION_HIT,ACTION_STAND):
    '''Initialising the policy for player
    '''
    POLICY_PLAYER = np.zeros(22, dtype=np.int)
    for i in range(12, 20):
            POLICY_PLAYER[i] = ACTION_HIT
    POLICY_PLAYER[20] = ACTION_STAND
    POLICY_PLAYER[21] = ACTION_STAND
    return POLICY_PLAYER


# In[3]:


def initialisation_blackjack():
    '''Initializing the parameters and vairables for blackjack
    '''
    # actions: hit or stand
    ACTION_HIT = 0
    ACTION_STAND = 1  #  "strike" in the book
    ACTIONS = [ACTION_HIT, ACTION_STAND]
    # policy for player
    POLICY_PLAYER  = policy_player_initialisation(ACTION_HIT,ACTION_STAND)
    return ACTION_HIT,ACTION_STAND,ACTIONS,POLICY_PLAYER


# In[ ]:





# In[4]:


def state_title(action_usable_ace, state_value_usable_ace,action_no_usable_ace, state_value_no_usable_ace):
    '''Function to give the value to the plots 
    
    '''
    images = []
    
    k = [action_usable_ace,
          state_value_usable_ace,
          action_no_usable_ace,
          state_value_no_usable_ace]
    l = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']
    for i in k:
        images.append(i)
    for i in l:
        titles.append(i)
        
    return images,titles


# In[5]:


def target_policy_player(usable_ace_player, player_sum, dealer_card):
    '''Function form of target policy of player'''
    
    return POLICY_PLAYER[player_sum]


# In[6]:


def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    '''Function form of behavior policy of player'''
    
    if np.random.binomial(1, 0.5) != 1:
        return ACTION_HIT
    return ACTION_STAND


# In[7]:


def policy_dealer_initialisation(ACTION_HIT,ACTION_STAND):
    '''Initialisation for policy of dealer 
    '''
    # policy for dealer
    POLICY_DEALER = np.zeros(22, dtype=np.int)
    min = 12
    max = 17
    max_l = 22
    for i in range(min, max):
        POLICY_DEALER[i] = ACTION_HIT
    for i in range(max, max_l):
        POLICY_DEALER[i] = ACTION_STAND
    return POLICY_DEALER


# In[ ]:





# In[8]:


def min_card(card):
    card = min(card, 10)
    return card
    
def get_card():
    ''' get a new card
    ''' 
    card = np.random.randint(1, 14)
    card = get_min(card)
    
    return card


# In[18]:


# get the value of a card (11 for ace).
def card_value(card_id):
    '''Function to return the card value
    Parameters are as: 
    card_id : Card id 
    
    '''
    if card_id == 1:
        return 11

    else:
        return card_id


# In[10]:


def condition_initial_state(initial_state,initial_action,policy_player,player_sum,
                                                                                      player_trajectory,usable_ace_player,dealer_card1,dealer_card2,usable_ace_dealer):
    '''Condition when initial_state is None or Not None
    '''
    if initial_state is not None:
            # generate a random initial state
            while player_sum < 12:
                # if sum of player is less than 12, always hit
                card = get_card()
                player_sum += card_value(card)
                # If the player's sum is larger than 21, he may hold one or two aces.
                if player_sum <= 21:
                    usable_ace_player |= (1 == card)
                    
                else:
                    assert player_sum == 22
                    # last card must be ace
                    player_sum -= 10
            # initialize cards of dealer, suppose dealer will show the first card he gets
            dealer_card1 = get_card()
            dealer_card2 = get_card()
    else:
        # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()
    return dealer_card1,dealer_card2,usable_ace_player, player_sum


# In[11]:


def compare_sum(player_sum,dealer_sum,state,player_trajectory):
    '''compare the sum between player and dealer
    Parameters are as : 
    player_sum : player sum
    dealer_sum : dealer sum
    state_player : state of palyer
    player_trajectory : Trajectory of player
    
    '''
    assert dealer_sum <= 21 and player_sum <= 21 
    
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory


# In[ ]:





# In[19]:





# In[13]:





# In[14]:





# In[15]:


# Monte Carlo Sample with Off-Policy
def monte_off_loop(episodes,behavior_policy_player,initial_state,rhos,returns):
    for i in range(0, episodes):
        _, reward, player_trajectory = play(behavior_policy_player, initial_state=initial_state)

        # get the importance ratio
        numerator = denominator =  1.0
        for (usable_ace, player_sum, dealer_card), action in player_trajectory:
            if action != target_policy_player(usable_ace, player_sum, dealer_card):
                numerator = 0.0
                break
            else:
                denominator *= 0.5
                
        rho = numerator / denominator
        rhos.append(rho)
        returns.append(reward)
    return rhos,returns

def monte_carlo_off_policy(episodes):
    initial_state = [True, 13, 2]

    rhos = []
    returns = []
    rhos,returns = monte_off_loop(episodes,behavior_policy_player,initial_state,rhos,returns)
    rhos = np.asarray(rhos)
    rhos = rhos.tolist()
    rhos = np.array(rhos)
    returns = np.asarray(returns)
    returns = returns.tolist()
    returns = np.array(returns)
    w_r = rhos * returns
    ####
    weighted_returns = copy.deepcopy(w_r) 
    weighted_returns = np.add.accumulate(weighted_returns)
    rhos = np.add.accumulate(rhos)
    arange_np = np.arange(1, episodes + 1)
    ordinary_sampling = weighted_returns / arange_np
    with np.errstate(divide='ignore',invalid='ignore'):
        check_w = weighted_returns / rhos
        w_s = np.where(rhos != 0,check_w, 0)
        weighted_sampling = copy.deepcopy(w_s)
    return ordinary_sampling, weighted_sampling


# In[16]:


def draw_fig(error_ordinary,error_weighted,num):
    plt.plot(error_ordinary, label='Ordinary Importance Sampling')
    plt.plot(error_weighted, label='Weighted Importance Sampling')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Mean square error')
    plt.xscale('log')
    plt.legend()
    
    plt.savefig('../images1/figure_5_'+str(num)+'.png')
    plt.close()


# In[ ]:





# In[ ]:





# In[17]:


# getting the initial parameters 
ACTION_HIT,ACTION_STAND,ACTIONS,POLICY_PLAYER = initialisation_blackjack()
POLICY_DEALER = policy_dealer_initialisation(ACTION_HIT,ACTION_STAND)

true_value = -0.27726
episodes = 10000
runs = 100
error_ordinary = np.zeros(episodes)
error_weighted = np.zeros(episodes)
for i in tqdm(range(0, runs)):
    ordinary_sampling_, weighted_sampling_ = monte_carlo_off_policy(episodes)
    # get the squared error
    power_val = np.power(ordinary_sampling_ - true_value, 2)
    error_ordinary =error_ordinary +power_val
    power_val = np.power(weighted_sampling_ - true_value, 2)
    error_weighted =error_weighted + power_val
error_ordinary =error_ordinary/runs
error_weighted = error_weighted/runs

draw_fig(error_ordinary,error_weighted,3)


# In[ ]:





# In[ ]:




