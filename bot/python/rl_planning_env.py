import numpy as np
import gym
import mappings
import sys
import os
import gzip
sys.path.append("build/bin")
import botlib_bindings
import simulator_visualizer

NUM_ACTIONS = 13
NUM_MINIMAPS = 5
MINIMAP_SIZE = 16
TENSOR_INPUT_SIZE = MINIMAP_SIZE*MINIMAP_SIZE*NUM_MINIMAPS + 5 * 5 + 6 + 35*5*2
print("Imported??")

def observeRLPlanningState(internalState):
    features = internalState.observe()
    result = np.concatenate([x.flatten() for x in features])
    assert result.shape == (TENSOR_INPUT_SIZE,), (result.shape, TENSOR_INPUT_SIZE)
    return result


def loadGzip(filepath):
    with gzip.open(filepath, 'rb') as f:
        return f.read()

binaryReplaysPath = "training_data/replays/b1"
binaryReplays = [os.path.join(binaryReplaysPath, x) for x in os.listdir(binaryReplaysPath)]
env_manager = botlib_bindings.RLEnvManager(simulator_visualizer, loadGzip, binaryReplays)

class RLPlanningEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        
        super(RLPlanningEnv, self).__init__()

        self.action_space = gym.spaces.discrete.Discrete(NUM_ACTIONS)
        # self.observation_space = gym.spaces.multi_discrete.MultiDiscrete([1 for _ in range(ACTIONS)] + [30])
        # self.observation_space = gym.spaces.multi_discrete.MultiDiscrete([30, 30, 30, 30, 30])

        self.observation_space = gym.spaces.box.Box(low=np.array([0] * TENSOR_INPUT_SIZE), high=np.array([1] * TENSOR_INPUT_SIZE), dtype=np.float32)

        self.internal_state = None
        self.steps = 0
        self.reset()
    
    def action_name(self, action):
        return self.internal_state.actionName(action)

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
        reward, done = self.internal_state.step(action)
        reward /= 100.0

        if self.steps > 800:
            done = True

        return observeRLPlanningState(self.internal_state), reward, done, {}

    def reset(self, gameStartConfig=False):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the
            space.
        """
        if gameStartConfig:
            assert False
        else:
            self.internal_state = env_manager.getEnv()
        
        self.steps = 0
        return observeRLPlanningState(self.internal_state)

    def render(self, mode='human', close=False):
        self.internal_state.print()
        return ""
    
    def visualization_info(self):
        return self.internal_state.visualizationInfo()
