import gym
from gym import spaces
import numpy as np
from temperature import get_next_temperature

class HeatingEnv(gym.Env):
    def __init__(self, df):
        super(HeatingEnv, self).__init__()
        self.env_name="heating_env"
        self.done = 0
        self.reward = 0
        # The heating system has two actions on/off
        self.action_space = spaces.Discrete(2)
        # state[0] we feel cold, we are ok (or nobody is in),
        # state[1] day of the week
        # state [2] hour of the day
        # state [4] minute of the hour
        # state [4] no interaction for more than 1 hour
        self.observation_space = spaces.MultiDiscrete([2,7,24,6])
        self.state=np.zeros(4)
        self.eec = 0
        self.time_step = 0
        self.initial_df=df
        # We discretize the system by every 1/6 hour
        self.granularity=1/6

    def reset(self):
        self.done = 0
        self.reward=0
        self.df=self.initial_df.copy()
        # suppose that initially , the system was on
        self.state[0]=0
        # the first day is a Monday
        self.state[1]=0
        self.state[2]=0
        self.state[3]=0
        self.time_step=0
        self.accumulated_no_interaction=0
        self.indoor_temperature = 18
        self.df["indoor_temperature"] = self.indoor_temperature
        self.df["is_heating_on"] = 0
        return self.state

    def step(self, action):
        if self.done == 1:
            print("end_of_learning")
        else:
            if self.time_step >= len(self.df) -1:
                self.done = 1
                print(self.reward)
            else:
                # compute the temperature of the next time step
                self.df.at[self.time_step,"is_heating_on"] = action
                self.df.at[self.time_step+1,"indoor_temperature"] = get_next_temperature(
                    action,
                    self.df.at[self.time_step,"indoor_temperature"],
                    self.df.at[self.time_step,"temperature"],
                    self.granularity)
                # With the action, get the next state
                self.time_step =self.time_step+1
                self.state[0] = int(self.df.at[self.time_step,"indoor_temperature"] < self.df.at[self.time_step,"lower_temperature"])

                self.state[1] = self.df.at[self.time_step,"time"].dayofweek
                self.state[2] = self.df.at[self.time_step,"time"].hour
                self.state[3] = self.df.at[self.time_step,"time"].minute/10

                # penalize if the system is on and people feel cold
                self.reward = self.reward -action - 1.2*self.state[0]
        info={}
        return self.state, self.reward, self.done, info

