import numpy as np
import random

class Qlearning:

    def __init__(self,env,alpha = 0.1,gamma = 0.6,epsilon = 0.1):
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon
        self.q_table=np.zeros((2,7,24,6,env.action_space.n))
        self.env=env

    def learn(self,total_timesteps=2e4):
        if self.env.env_name=="heating_env":
            state = self.env.reset()
            for i in range(int(total_timesteps)):
                done = 0

                while done ==0:
                    state = list(map(int, state))
                    if random.uniform(0, 1) < self.epsilon:
                        action = self.env.action_space.sample()  # Explore action space
                    else:
                        action = np.nanargmax(self.q_table[state[0],state[1],state[2],state[3],:])  # Exploit learned values
                    next_state, reward, done, info = self.env.step(action)
                    next_state=list(map(int,next_state))
                    old_value = self.q_table[state[0],state[1],state[2],state[3],action]
                    next_max = np.max(self.q_table[next_state[0],next_state[1],next_state[2],next_state[3],:])
                    new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                    self.q_table[state[0],state[1],state[2],state[3], action] = new_value
                    state = next_state
                state = self.env.reset()
        return self.q_table


    def predict(self,state,deterministic=True):
        state= list(map(int, state))
        action=np.argmax(self.q_table[state[0],state[1],state[2],state[3],:])
        if (~deterministic) & (random.uniform(0, 1) < self.epsilon):
            action = self.env.action_space.sample()
        return action