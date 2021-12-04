import gym
from gym import spaces
import pandas as pd
import numpy as np


def unscale_interest_rate(scaled_interest_rate):
    return (scaled_interest_rate - 1) * 0.002 + 0.002


class InterestEnv(gym.Env):

    def render(self, mode="human"):
        pass

    def __init__(self, config):
        """
        Inits the environment. The configuration is passed to the constructor via the env_config parameter
        in the Trainer constructor or tune.run method.
        :param config:
        """
        # file destination of the interest rates
        self.product_path = config["product_path"]
        # determines the number of interest rates seen by the agent at a certain time step
        self.window_length = config["window_length"]
        self.start_timestamp = pd.to_datetime(config["start_timestamp"]) if "start_timestamp" in config else None
        self.end_timestamp = pd.to_datetime(config["end_timestamp"]) if "end_timestamp" in config else None

        print("Calling init env with product path: {} \n window-length: {} \n start_timestamp: {} \n end_timestamp: {}"
              .format(self.product_path,
                      self.window_length,
                      self.start_timestamp,
                      self.end_timestamp))

        self.observations = None
        self.current_value = np.array([1.0], dtype=np.float32)
        self.current_step = 0
        self.transaction_fees = 0.075 / 100
        self.interest_rates = None

        # can be between -1 (= 100% short) and +1 (= 100% long)
        self.current_position = 0

        # Define actions: Buy or sell or hold
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=(1,),
                                       dtype=np.float32)

        # Example input is a matrix of (window_length, 2 feature)
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(self.window_length,
                                                   # interest_rate
                                                   1),
                                            dtype=np.float32)
        self.create_observations()

    def create_observations(self):
        print("Creating observations for product: {}".format(self.product_path))

        # read the data and process it
        interest_rates = pd.read_csv(self.product_path)
        interest_rates['timestamp'] = pd.to_datetime(interest_rates['timestamp'])
        interest_rates = interest_rates.set_index("timestamp", drop=True).sort_index()

        # normalize to [-1; 1] values
        # maximum interest rate for 4h is +/-0.2% (+/-0.05% p.h.)
        interest_rates['normalized_interest_rate'] = (interest_rates['interest_rate'] - 0.002) / 0.002 + 1

        # range the interest rates if the start timestamp is set
        if self.start_timestamp is not None:
            interest_rates = interest_rates[interest_rates.index >= self.start_timestamp]

        # range the interest rate if the end timestamp is set
        if self.end_timestamp is not None:
            interest_rates = interest_rates[interest_rates.index < self.end_timestamp]

        self.normalized_interest_rates = interest_rates['normalized_interest_rate'].values
        self.interest_rates = interest_rates['interest_rate'].values
        self.timestamps = interest_rates.index.values
        self.df_interest_rates = interest_rates

    def unscale_interest_rate(self, scaled_interest_rate):
        return (scaled_interest_rate - 1) * 0.002 + 0.002

    def step(self, action):
        pos_0 = self.current_position.copy()
        value_0 = self.current_value.copy()

        # calculate the weight change based on the current position
        weight_change = action - pos_0

        scaled_interest_rate = self.normalized_interest_rates[self.current_step + self.window_length]
        # unscale the the current scaled interest rate
        unscaled_interest_rate = unscale_interest_rate(scaled_interest_rate)
        # add the realized interest rate to the current value
        self.current_value *= 1 + unscaled_interest_rate * pos_0

        fees = np.abs(weight_change * self.transaction_fees)

        # reduce current value by fees
        self.current_value *= 1 - fees
        self.current_position = action
        self.current_step += 1

        # Calculate the reward
        if self.current_value < 0:
            reward = np.log(value_0)
            done = True
        else:
            # reward as log return of the portfolio value
            reward = (np.log(self.current_value) - np.log(value_0))
            # reward = self.current_value - value_0
            done = (self.current_step + self.window_length >= len(self.normalized_interest_rates) - 1)

        info = {
            "timestamp": self.timestamps[self.current_step + self.window_length],
            "value": self.current_value.copy(),
            "interest_rate": self.interest_rates[self.current_step + self.window_length],
            "position": self.current_position.copy()
        }

        observation = np.zeros(shape=(self.window_length, 1))

        # create observation with weight set to zero and interest rates
        observation[0:self.window_length, 0] = self.normalized_interest_rates[
                                               self.current_step:self.window_length + self.current_step]

        observation = observation[np.newaxis, np.newaxis, :]

        return observation, reward, done, info

    def reset(self):
        self.current_step = 0
        self.current_value = np.array([1], dtype=np.float32)
        self.current_position = np.array([0], dtype=np.float32)
        observation = np.zeros(shape=(self.window_length, 1))
        # create observation with weight set to zero and interest rates
        observation[0:self.window_length, 0] = self.normalized_interest_rates[0:self.window_length]
        observation = observation[np.newaxis, np.newaxis, :, :]
        return observation

    def close(self):
        pass
