import os
import random
import math

import numpy as np
import gym.spaces.discrete
import gym
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from custom_a2c import CustomA2C


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (0, 1)

    def __init__(self):
        super(CustomEnv, self).__init__()

        self.action_space = gym.spaces.discrete.Discrete(4)
        # self.observation_space = gym.spaces.multi_discrete.MultiDiscrete([1 for _ in range(ACTIONS)] + [30])
        # self.observation_space = gym.spaces.multi_discrete.MultiDiscrete([30, 30, 30, 30, 30])
        self.observation_space = gym.spaces.box.Box(low=np.array([0] * 125), high=np.array([30] * 125), dtype=np.float32)

        self.internal_state = 0
        self.ticks = 0
        self.k = 0
        self.did_tick_time = False
        self.reset()

    def _observation_model(self, internal_state):
        m = np.array(internal_state)
        m[4] = max(5*m[4] - (m[3]+m[2]), 0)
        m = np.minimum(np.maximum(m, 0), 29)

        m2 = np.zeros(121 + 4)
        m2[0] = m[0]
        m2[1] = m[1]
        m2[2] = m[2]
        m2[3] = m[3]
        m2[4] = m[4]

        m2[5+30*0:5+30*0+m[1]+1] = 1
        m2[5+30*1:5+30*1+m[2]+1] = 1
        m2[5+30*2:5+30*2+m[3]+1] = 1
        m2[5+30*3:5+30*3+m[4]+1] = 1
        return m2

    def _observation_model2(self, internal_state):
        m = np.array(internal_state)
        m[4] = max(5*m[4] - (m[3]+m[2]), 0)
        m = np.minimum(np.maximum(m, 0), 29)

        m2 = np.zeros(150)
        m2[30*0+m[0]] = 1
        m2[30*1+m[1]] = 1
        m2[30*2+m[2]] = 1
        m2[30*3+m[3]] = 1
        m2[30*4+m[4]] = 1

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # print(action)
        reward = 0

        while self.internal_state[2] > 0 and self.did_tick_time:
            self.internal_state[2] -= 1
            self.internal_state[1] += 1
            self.internal_state[3] += 1
            reward += 1

        self.did_tick_time = False

        if action == 1:
            # Build some building
            self.internal_state[1] += 1

            self.did_tick_time = True
        elif action == 2:
            if self.internal_state[1] > 0 and self.internal_state[3]+self.internal_state[2] < 5*self.internal_state[4]:
                self.internal_state[1] -= 1
                self.internal_state[2] += 1
            else:
                self.did_tick_time = True
        elif action == 3:
            self.internal_state[4] += 1

            self.did_tick_time = True
        else:
            self.did_tick_time = True

        if self.did_tick_time:
            # Increment time
            self.internal_state[0] += 1

        # reward = 1 if action == 3 else 0
        done = self.internal_state[0] >= 30

        return self._observation_model(self.internal_state), reward, done, {'r': reward}

    def reset(self, randomState=True):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        self.internal_state = [0, random.randrange(5), random.randrange(5), random.randrange(5), random.randrange(5)]
        if not randomState:
            for i in range(len(self.internal_state)):
                self.internal_state[i] = 0

        self.ticks = 0
        self.did_tick_time = True
        return self._observation_model(self.internal_state)

    def render(self, mode='human', close=False):
        return f"state: {self.internal_state}, ticks: {self.ticks}"