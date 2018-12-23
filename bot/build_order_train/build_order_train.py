import json
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'next_action', 'reward'))

TERRAN_REAPER = 49


unitIndexMap = {
    29: 0,  # TERRAN_ARMORY
    55: 1,  # TERRAN_BANSHEE
    21: 2,  # TERRAN_BARRACKS
    38: 3,  # TERRAN_BARRACKSREACTOR
    37: 4,  # TERRAN_BARRACKSTECHLAB
    57: 5,  # TERRAN_BATTLECRUISER
    24: 6,  # TERRAN_BUNKER
    18: 7,  # TERRAN_COMMANDCENTER
    692: 8,  # TERRAN_CYCLONE
    22: 9,  # TERRAN_ENGINEERINGBAY
    27: 10,  # TERRAN_FACTORY
    40: 11,  # TERRAN_FACTORYREACTOR
    39: 12,  # TERRAN_FACTORYTECHLAB
    30: 13,  # TERRAN_FUSIONCORE
    50: 14,  # TERRAN_GHOST
    26: 15,  # TERRAN_GHOSTACADEMY
    53: 16,  # TERRAN_HELLION
    484: 17,  # TERRAN_HELLIONTANK
    689: 18,  # TERRAN_LIBERATOR
    51: 19,  # TERRAN_MARAUDER
    48: 20,  # TERRAN_MARINE
    54: 21,  # TERRAN_MEDIVAC
    23: 22,  # TERRAN_MISSILETURRET
    268: 23,  # TERRAN_MULE
    132: 24,  # TERRAN_ORBITALCOMMAND
    130: 25,  # TERRAN_PLANETARYFORTRESS
    56: 26,  # TERRAN_RAVEN
    49: 27,  # TERRAN_REAPER
    20: 28,  # TERRAN_REFINERY
    45: 29,  # TERRAN_SCV
    25: 30,  # TERRAN_SENSORTOWER
    32: 31,   # TERRAN_SIEGETANKSIEGED
    33: 31,  # TERRAN_SIEGETANK
    28: 32,  # TERRAN_STARPORT
    42: 33,  # TERRAN_STARPORTREACTOR
    41: 34,  # TERRAN_STARPORTTECHLAB
    19: 35,  # TERRAN_SUPPLYDEPOT
    52: 36,  # TERRAN_THOR
    691: 36, # TERRAN_THORAP
    34: 37,   # TERRAN_VIKINGASSAULT
    35: 37,  # TERRAN_VIKINGFIGHTER
    498: 38,  # TERRAN_WIDOWMINE,
}

isUnitMilitary = {
    29: False,  # TERRAN_ARMORY
    55: True,  # TERRAN_BANSHEE
    21: False,  # TERRAN_BARRACKS
    57: True,  # TERRAN_BATTLECRUISER
    24: False,  # TERRAN_BUNKER
    18: False,  # TERRAN_COMMANDCENTER
    692: True,  # TERRAN_CYCLONE
    22: False,  # TERRAN_ENGINEERINGBAY
    27: False,  # TERRAN_FACTORY
    30: False,  # TERRAN_FUSIONCORE
    50: True,  # TERRAN_GHOST
    26: False,  # TERRAN_GHOSTACADEMY
    53: True,  # TERRAN_HELLION
    484: True,  # TERRAN_HELLIONTANK
    689: True,  # TERRAN_LIBERATOR
    51: True,  # TERRAN_MARAUDER
    48: True,  # TERRAN_MARINE
    54: True,  # TERRAN_MEDIVAC
    23: False,  # TERRAN_MISSILETURRET
    132: False,  # TERRAN_ORBITALCOMMAND
    130: False,  # TERRAN_PLANETARYFORTRESS
    56: True,  # TERRAN_RAVEN
    49: True,  # TERRAN_REAPER
    20: False,  # TERRAN_REFINERY
    45: False,  # TERRAN_SCV
    25: False,  # TERRAN_SENSORTOWER
    32: True,   # TERRAN_SIEGETANKSIEGED
    33: True,  # TERRAN_SIEGETANK
    28: False,  # TERRAN_STARPORT
    19: False,  # TERRAN_SUPPLYDEPOT
    52: True,  # TERRAN_THOR
    34: True,   # TERRAN_VIKINGASSAULT
    35: True,  # TERRAN_VIKINGFIGHTER
    498: True,  # TERRAN_WIDOWMINE
}

NUM_UNITS = len(set(unitIndexMap.values()))
TENSOR_INPUT_SIZE = NUM_UNITS * 3 + 6
NUM_ACTIONS = NUM_UNITS
MILITARY_UNITS_MASK = np.zeros(NUM_UNITS)
for k, v in unitIndexMap.items():
    MILITARY_UNITS_MASK[v] = 1 if isUnitMilitary[k] else 0

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.durations = []
        self.health_diffs = []
        self.total_rewards = []
        self.all_actions = []
        self.max_error = 1000
        self.prioritized_replay = True

        if self.prioritized_replay:
            self.error_buckets = []
            self.priority_samples = []
            for i in range(10):
                self.error_buckets.append([])

            self.count = 0

    def push(self, transition: Transition):
        """Saves a transition."""
        if self.prioritized_replay:
            random.choice(self.error_buckets).append(transition)
            self.count += 1
        else:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.prioritized_replay:
            result = []
            while len(result) < batch_size and len(self.priority_samples) > 0:
                result.append(self.priority_samples.pop())

            changed = True
            while len(result) < batch_size and changed:
                changed = False
                for bucket in self.error_buckets:
                    # Take N samples from each bucket according to the log2 of their size.
                    n = min(len(bucket), int(math.log2(max(2,len(bucket)))))
                    for i in range(n):
                        changed = True
                        idx = random.randrange(0, len(bucket))
                        result.append(bucket[idx])
                        bucket[idx] = bucket[-1]
                        bucket.pop()
                        self.count -= 1
            return result
        else:
            return random.sample(self.memory, batch_size)

    def discard_random(self):
        if not self.prioritized_replay:
            assert False
        idx = random.randrange(0, self.count)
        for bucket in self.error_buckets:
            if idx < len(bucket):
                bucket[idx] = bucket[-1]
                bucket.pop()
                self.count -= 1
                return
            idx -= len(bucket)

        assert False, (idx, self.count, sum(map(len, self.error_buckets)))

    def insert(self, samples, errors):
        if not self.prioritized_replay:
            return

        assert len(samples) == len(errors)
        for i in range(len(samples)):
            # Discard a random sample
            while self.count >= self.capacity:
                self.discard_random()

            bucket_idx = max(0, min(len(self.error_buckets)-1, math.floor(len(self.error_buckets) * errors[i].item() / self.max_error)))
            self.error_buckets[bucket_idx].append(samples[i])
            self.count += 1

    def createState(self, state, goalTensor):
        STRIDE = 3
        inputTensor = torch.zeros(TENSOR_INPUT_SIZE, dtype=torch.float)
        for unit in state["units"]:
            unitIndex = unitTypeNameToIndex[unit["type"]]
            if unit["progress"] < 1:
                # In progress
                inputTensor[NUM_UNITS * 1 + unitIndex] += 1
            else:
                # Finished
                inputTensor[NUM_UNITS * 0 + unitIndex] += 1
        
        inputTensor[NUM_UNITS*2:] = goalTensor

        # Some metadata, the data is normalized to approximately 1
        inputTensor[NUM_UNITS*STRIDE + 0] = state["minerals"] / 100
        inputTensor[NUM_UNITS*STRIDE + 1] = state["vespene"] / 100
        inputTensor[NUM_UNITS*STRIDE + 2] = state["remainingFood"] / 10
        inputTensor[NUM_UNITS*STRIDE + 3] = state["mineralsPerSecond"] / 10
        inputTensor[NUM_UNITS*STRIDE + 4] = state["vespenePerSecond"] / 10
        inputTensor[NUM_UNITS*STRIDE + 5] = state["mineralsInLowestBase"] / 1000

        return inputTensor
    
    def createGoalTensor(self, goal):
        inputTensor = torch.zeros(NUM_UNITS, dtype=torch.float)
        for unit in goal["units"]:
            unitIndex = unitTypeNameToIndex[unit["type"]]
            inputTensor[unitIndex] += unit["weight"]
        
        inputTensor /= inputTensor.sum()
        return inputTensor

    def calculate_reward(self, s1, s2, goalTensor, deltaTime):
        s1unitCounts = s1.numpy()[0:NUM_UNITS]
        s2unitCounts = s2.numpy()[0:NUM_UNITS]
        # How many units were added
        deltaUnitCounts = np.max(0, s2unitCounts - s1unitCounts)
        # Total number of military units
        numMilitaryUnits = (s1unitCounts * MILITARY_UNITS_MASK).sum()
        # Number of units that we do want
        desiredUnitCounts = goalTensor * np.max(numMilitaryUnits, 1)

        falloff = 0.2
        scorePerUnit = np.min(1.0, np.exp((desiredUnitCounts - s1unitCounts)*falloff))

        # Get a score if we added a unit of that type
        reward = (deltaUnitCounts * scorePerUnit).sum()

        # Assume the reward happens right before s2.
        # If the build order involved satisfying some implicit constraints or maybe some waiting time
        # then the reward will be rewarded right at the end.
        # This makes it beneficial for the agent to learn to handle implicit dependencies by itself, but it can still fall back on them without too big of a loss.
        reward *= math.pow(GAMMA_PER_SECOND, deltaTime)
        return reward, terminal_state

    def determine_action(self, s1, s2, unit):
        pass

    def loadSession(self, s):
        data = json.loads(s)
        # print(s)
        states = data["states"]
        goalTensor = createGoalTensor(data["goal"])
        total_reward = 0

        tensor_states = [self.createState(s, goal) for s in states]

        for i in range(len(tensor_states)-1):
            s1 = states[i]
            s2 = states[i+1]
            a1 = states[i]["action"]
            a2 = states[i+1]["action"]

            reward, terminal_state = self.calculate_reward(s1, s2, goalTensor)

            total_reward += reward
            self.push(Transition(state=s1, action=a1, next_state=s2, reward=reward, next_action=a2))

        self.total_rewards.append(total_reward)
        self.durations.append(len(states))

    def __len__(self):
        return self.count if self.prioritized_replay else len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        layers = []
        layers.append(nn.Linear(TENSOR_INPUT_SIZE, 100))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(100, 40))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(40, 40))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(40, 40))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(40, NUM_ACTIONS))
        self.seq = nn.Sequential(*layers)

    def forward(self, inputTensor):
        c = self.seq(c)
        return x  # B x NUM_ACTIONS


lastState = None


def get_reward(s, unitTags):
    global lastState
    state = json.loads(s)
    # print(state)

    rewards = []
    for tag in unitTags:
        unit_in_last_state = None
        for u in lastState["units"]:
            if u["tag"] == tag:
                unit_in_last_state = u

        if unit_in_last_state is not None:
            reward, _ = memory.calculate_reward(lastState, state, unit_in_last_state)
        else:
            reward = 0
        rewards.append(float(reward))

    lastState = state
    return rewards


def predict(s, unitTags, enableExploration):
    state = json.loads(s)
    # print(state)

    actions = []
    for tag in unitTags:
        unit = None
        for u in state["units"]:
            if u["tag"] == tag:
                unit = u
                break

        # print(f"Time since attacked: {unit['last_attacked_tick'] - state['tick']}")
        assert unit is not None, "unit did not exist in state"
        t1 = memory.createState(unit, state)
        # print(t1)

        action = select_action(t1, enableExploration)
        actions.append(action)

    return actions


def addSession(s):
    index = len(os.listdir(data_path))
    with open(data_path + "/session" + str(index) + ".json", "w") as f:
        f.write(s)

    memory.loadSession(s)
    pass


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


def load_all(optimization_steps_per_load: int):
    print("Loading training data...")
    fs = os.listdir(data_path)
    fs = natural_sort(fs)
    # random.shuffle(fs)
    for p in fs:
        f = open(data_path + "/" + p)
        s = f.read()
        f.close()
        memory.loadSession(s)
        optimize(optimization_steps_per_load)
    print("Done")


data_path = "training_data/build_order/3"
BATCH_SIZE = 32
GAMMA_PER_SECOND = 0.96
TICKS_PER_STATE = 10
TICKS_PER_SECOND = 22.4
GAMMA = math.pow(GAMMA_PER_SECOND, TICKS_PER_STATE / TICKS_PER_SECOND)
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000
TEMPERATURE_START = 10
TEMPERATURE_END = 1
TEMPERATURE_DECAY = 8000
TARGET_UPDATE = 1

NEARBY_UNIT_DISTANCE_THRESHOLD = 10
NEARBY_ALLY_UNIT_DISTANCE_THRESHOLD = 50

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayMemory(5000)


steps_done = 0


def eps():
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)


def temperature():
    return TEMPERATURE_END + (TEMPERATURE_START - TEMPERATURE_END) * math.exp(-steps_done/TEMPERATURE_DECAY)


def select_action(state, enableExploration):
    global steps_done
    eps_threshold = eps()
    sample = random.random()

    steps_done += 1

    with torch.no_grad():
        policy_net.eval()
        q = policy_net(state.unsqueeze(0))
        temp = temperature() if enableExploration else 0.1

        soft = q[0].numpy()
        soft = np.exp((soft - soft.max())/temp)
        soft = soft/np.sum(soft)

        action = np.random.choice(NUM_ACTIONS, p=soft)
        mx = q[0][action].item()
        # mx = q.max(1)
        print(f"Expected reward: {mx}")
        policy_net.train()

    if sample > eps_threshold or not enableExploration:
        pass
    else:
        action = random.randrange(NUM_ACTIONS)

    return (action, list(q[0].numpy()), list(state[0].numpy()), list(state[1].numpy()), list(state[2].numpy()))


episode_durations = []
losses = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    policy_net.train()
    transitions = memory.sample(BATCH_SIZE)
    batch_size = len(transitions)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)

    # No non-final states, will cause some torch errors
    any_non_final = len([s for s in batch.next_state if s is not None]) > 0
    if any_non_final > 0:
        non_final_next_states = torch.cat([s[0].unsqueeze(0) for s in batch.next_state if s is not None])

    state_batch = torch.cat([s[0].unsqueeze(0) for s in batch.state])
    # assert state_batch.size() == (batch_size,)

    action_batch = torch.tensor(batch.action).unsqueeze(1)
    assert(action_batch.size() == (batch_size,1))
    reward_batch = torch.tensor(batch.reward, dtype=torch.float)
    assert reward_batch.size() == (batch_size,)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch0, state_batch1, state_batch2).gather(1, action_batch)
    assert state_action_values.size() == (batch_size, 1)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(batch_size, device=device)
    if any_non_final:
        next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()

    assert next_state_values.size() == (batch_size,)

    # torch.set_printoptions(threshold=10000)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    assert expected_state_action_values.size() == (batch_size,)

    # Compute Huber loss
    transition_losses = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
    loss = transition_losses.mean()
    print(loss.item())
    losses.append(loss.item())
    memory.insert(transitions, transition_losses)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
        # param.grad.data.clamp_(-1, 1)
    optimizer.step()


plt.ioff()
episode = 0

epss = []
temps = []

def plot_loss():
    durations_t = torch.tensor(losses, dtype=torch.float)
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(2,3,1)
    plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    ax.set_yscale('log')
    plt.plot(durations_t.numpy())


    x = torch.tensor(memory.health_diffs, dtype=torch.float)
    smooth = np.convolve(x.numpy(), np.ones((10,))/10, mode='valid')
    ax = fig.add_subplot(2,3,2)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Health Diff')
    plt.plot(x.numpy())
    plt.plot(smooth)

    x = torch.tensor(memory.total_rewards, dtype=torch.float)
    smooth = np.convolve(x.numpy(), np.ones((10,))/10, mode='valid')
    ax = fig.add_subplot(2,3,3)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(x.numpy())
    plt.plot(smooth)

    ax = fig.add_subplot(2,3,4)
    # Use the last 50% of the actions
    action_slice = memory.all_actions[len(memory.all_actions)//2:]
    counts = np.bincount(action_slice)
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Action')
    plt.ylabel('Times Chosen')


    if memory.prioritized_replay:
        ax = fig.add_subplot(2,3,5)
        counts = [len(bucket) for bucket in memory.error_buckets]
        plt.bar(range(len(counts)), counts)
        plt.xlabel('Memory Bucket')
        ax.set_yscale('log')
        plt.ylabel('Size')

    ax = fig.add_subplot(2,3,6)
    epss.append(eps())
    temps.append(temperature())

    plt.plot(epss, label="Epsilon")
    plt.plot(temps, label="Temperature")
    plt.xlabel('Time')
    plt.ylabel('Value')

    plt.pause(0.001)  # pause a bit so that plots are updated



def optimize(steps: int):
    global episode
    for i in range(steps):
        optimize_model()
        episode += 1
        # if episode % TARGET_UPDATE == 0:
        #     target_net.load_state_dict(policy_net.state_dict())

    plot_loss()
    # plt.show()

testSession = """
{
    "states": [
        {
            "tick": 0,
            "playerID": 1,
            "units": [],
            "walkableMap": []
        }
    ]
}"""

addSession(testSession)

if __name__ == "__main__":
    load_all(20)
    while True:
        optimize(200)
else:
    load_all(200)
    pass
