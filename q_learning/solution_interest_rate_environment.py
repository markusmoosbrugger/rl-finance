import gym
from gym import spaces
import pandas as pd
import numpy as np


class InterestRateEnv(gym.Env):
    """ OpenAI Gym Base Interest Rate Environment

    Attributes
    ----------
    action_space: gym.spaces.Discrete
        Agent's action space
    observation_space: gym.spaces.Discrete
        Agent's observation space


    Methods
    -------
    step(action)
        The agent takes a step in the environment
    reset()
        Resets the state of the environment and returns an initial observation
    render()
        Renders/displays the environment
    close()
        Performs any necessary cleanup

    """

    def __init__(self, product_path, window_length=5, start=None, end=None):
        super(InterestRateEnv, self).__init__()
        
        # Initialization
        self.current_step = 0
        self.window_length = window_length
        self.product_path = product_path
        self.transaction_fees = 0.075/100
        self.rounding_value = 0.05
        
        # Portfolio value
        self.current_value = 1.0
             
        num_states = int(2/self.rounding_value)
        num_actions = 3
        self.current_position = 0
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)
        
        self.create_interest_rates(start, end)

    def create_interest_rates(self, start, end):
        interest_rates = pd.read_csv(self.product_path)
        interest_rates['timestamp'] = pd.to_datetime(interest_rates['timestamp'])
        
        # filter interest rates between specified start and end
        if start:
            interest_rates = interest_rates[interest_rates['timestamp'] > start]                              
        if end:
            interest_rates = interest_rates[interest_rates['timestamp'] < end]
            
        # normalize to values in interval [-1, 1]
        # maximum interest rate for 4h is +/-0.2% (+/-0.05% per hour)
        interest_rates['normalized_interest_rate'] = (interest_rates['interest_rate'] - 0.002) / 0.002 + 1
       
        # use rolling window to calculate average interest rate within last elements
        interest_rates['normalized_interest_rate_avg'] = interest_rates['normalized_interest_rate'].rolling(window=self.window_length, center=False).mean()
        
        # remove entries which do not have an average value from dataframe
        interest_rates = interest_rates.dropna()
        
        # assign interest rate values to env object
        self.interest_rates_avg = interest_rates['normalized_interest_rate_avg'].values
        self.interest_rates = interest_rates['interest_rate'].values
        self.timestamps = interest_rates['timestamp'].values
        self.df_interest_rates = interest_rates
   
    # applies an action in the environment; classic agent-environment loop
    def step(self, action):
        pos_0 = self.current_position
        value_0 = self.current_value
        
        # convert action from [0, 2] to [-1, 1]
        action -= 1
        
        # check if weight has changed and calculate fees
        weight_change = action - pos_0
        fees = abs(weight_change) * self.transaction_fees
        
        # TODO: calculate new value with current interest rate and fees
        self.current_value = self.current_value * (1 - fees)
        interest_rate = self.interest_rates[self.current_step]
        self.current_value = self.current_value * (1 + interest_rate * pos_0)
    
        # TODO: calculate reward as log return of the portfolio values (formula on slide Reward)
        reward = np.log(self.current_value) - np.log(value_0)
        
        # Update position, timestep, check if done and add values to info
        self.current_position = action
        self.current_step += 1

        done = self.current_step >= len(self.interest_rates_avg) - 1

        info = {
            "timestamp": self.timestamps[self.current_step],
            "value": self.current_value,
            "position": self.current_position,
            "interest_rate": interest_rate
        }
        
        # Get observation with current avg interest rate
        normalized_interest_rate_avg = self.interest_rates_avg[self.current_step]
        observation = self.get_state_for_interest_rate(normalized_interest_rate_avg)
        
        return observation, reward, done, info

    # resets the environment
    def reset(self):

        self.current_step = 0
        self.current_value = 1.0
        self.current_position = 0
        observation = self.get_state_for_interest_rate(0)
        
        return observation 
    
    # rounds to precision a, and then to two significant digits
    def round_nearest(self,x, a):
        return round(round(x / a) * a, 2)
    
    # returns a discrete state given the interest rate
    def get_state_for_interest_rate(self, interest_rate):
        state = int((interest_rate + 1)/self.rounding_value)
        if state == self.observation_space.n:
            # if funding rate is 1.0 we return state 39 as state 40 does not exist
            state -= 1
        return state
      
    # returns a rounded interest rate given a discrete state
    def get_interest_rate_for_state(self, state):
        return round(state * self.rounding_value -1, 2)

    # returns the maximum number of steps per episode
    def maximum_episode_steps(self):
        return len(self.df_interest_rates)

    # prints the current portfolio value
    def render(self, mode='human'):
        print("Current value: {:.3f}".format(self.current_value))

    def close(self):
        pass
