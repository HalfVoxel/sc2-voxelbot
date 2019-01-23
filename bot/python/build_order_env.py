import numpy as np
import gym
import json
from build_order_loader import BuildOrderLoader, TENSOR_INPUT_SIZE, NUM_UNITS, reverseUnitIndexMap
import sys
sys.path.append("build/bin")
from cppbot import EnvironmentState

NUM_ACTIONS = NUM_UNITS
print("Imported??")

class BuildOrderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        
        super(BuildOrderEnv, self).__init__()

        self.action_space = gym.spaces.discrete.Discrete(NUM_ACTIONS)
        # self.observation_space = gym.spaces.multi_discrete.MultiDiscrete([1 for _ in range(ACTIONS)] + [30])
        # self.observation_space = gym.spaces.multi_discrete.MultiDiscrete([30, 30, 30, 30, 30])

        self.observation_space = gym.spaces.box.Box(low=np.array([0] * TENSOR_INPUT_SIZE), high=np.array([1] * TENSOR_INPUT_SIZE), dtype=np.float32)

        self.internal_state = EnvironmentState()
        self.loader = BuildOrderLoader(gamma_per_second=0.99)
        self.steps = 0
        self.reset()

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

        self.steps += 1
        t1 = self.loader.createState(json.loads(self.internal_state.getState()))
        time1 = self.internal_state.getTime()
        unit_type = reverseUnitIndexMap[action]
        self.internal_state.step(unit_type)
        t2 = self.loader.createState(json.loads(self.internal_state.getState()))
        time2 = self.internal_state.getTime()
        goal = self.loader.createGoalTensor(json.loads(self.internal_state.getGoal()))

        reward = self.loader.calculate_reward(t1, t2, goal, time2 - time1)
        observation = self.loader.combineStateAndGoal(t2, goal)
        done = time2 > 60*6 or self.steps > 60

        return observation, reward, done, {}

    def reset(self, gameStartConfig=False):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        if gameStartConfig:
            self.internal_state.resetToGameStartConfig()
        else:
            self.internal_state.reset()
        
        self.steps = 0
        state = self.loader.createState(json.loads(self.internal_state.getState()))
        goal = self.loader.createGoalTensor(json.loads(self.internal_state.getGoal()))
        observation = self.loader.combineStateAndGoal(state, goal)
        return observation

    def render(self, mode='human', close=False):
        self.internal_state.print()
        return ""