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
# Fix crash bug on some macOS versions
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'next_action', 'reward', 'deltaTime'))

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
TENSOR_INPUT_SIZE = NUM_UNITS * 4 + 7
NUM_ACTIONS = NUM_UNITS
MILITARY_UNITS_MASK = np.zeros(NUM_UNITS)

for k, v in unitIndexMap.items():
    MILITARY_UNITS_MASK[v] = 1 if k in isUnitMilitary and isUnitMilitary[k] else 0

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.durations = []
        self.health_diffs = []
        self.total_rewards = []
        self.all_actions = []
        self.max_error = 8
        self.prioritized_replay = False
        self.goalPool = []

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

    def createState(self, state):
        STRIDE = 3
        unitCounts = torch.zeros(NUM_UNITS, dtype=torch.float)
        unitsAvailable = torch.zeros(NUM_UNITS, dtype=torch.float)
        unitsInProgress = torch.zeros(NUM_UNITS, dtype=torch.float)
        metaTensor = torch.zeros(7, dtype=torch.float)        

        for unit in state["units"]:
            # Finished
            # TODO: Addon
            unitIndex = unitIndexMap[unit["type"]]
            unitCounts[unitIndex] += unit["totalCount"]
            unitsAvailable[unitIndex] += unit["availableCount"]

        for unit in state["unitsInProgress"]:
            # In progress
            unitIndex = unitIndexMap[unit["type"]]
            unitsInProgress[unitIndex] += 1

        # Some metadata, the data is normalized to approximately 1
        metaTensor[0] = state["minerals"] / 100
        metaTensor[1] = state["vespene"] / 100
        metaTensor[2] = state["foodAvailable"] / 10 if "foodAvailable" in state else 0
        metaTensor[3] = state["mineralsPerSecond"] / 10
        metaTensor[4] = state["vespenePerSecond"] / 10
        metaTensor[5] = state["highYieldMineralSlots"] / 10 if "highYieldMineralSlots" in state else 0
        metaTensor[6] = state["lowYieldMineralSlots"] / 10 if "lowYieldMineralSlots" in state else 0

        stateTensor = torch.cat([unitCounts, unitsAvailable, unitsInProgress, metaTensor])
        return stateTensor

    def combineStateAndGoal(self, stateTensor, goalTensor):
        assert(goalTensor.size() == (NUM_UNITS,))
        inputTensor = torch.cat([stateTensor, goalTensor])
        assert(inputTensor.size() == (TENSOR_INPUT_SIZE,))
        return inputTensor
    
    def createGoalTensor(self, goal):
        inputTensor = torch.zeros(NUM_UNITS, dtype=torch.float)
        for unit in goal:
            unitIndex = unitIndexMap[unit["type"]]
            inputTensor[unitIndex] += unit["count"]
            assert(unit["count"] >= 0)
        
        s = inputTensor.sum()
        if s > 0:
            inputTensor /= s
        return inputTensor

    def calculate_reward(self, t1, t2, goalTensor, deltaTime):
        s1unitCounts = t1.numpy()[0:NUM_UNITS]
        s2unitCounts = t2.numpy()[0:NUM_UNITS]
        # How many units were added
        deltaUnitCounts = np.maximum(0, s2unitCounts - s1unitCounts)
        # Total number of military units
        numMilitaryUnits = (s1unitCounts * MILITARY_UNITS_MASK).sum()
        # Number of units that we do want
        desiredUnitCounts = goalTensor.numpy() * np.maximum(numMilitaryUnits, 1)

        falloff = 0.2
        # TODO: multiply by resource cost
        # scorePerUnit = np.minimum(1.0, np.exp((desiredUnitCounts - s1unitCounts)*falloff))
        scorePerUnit = np.zeros(NUM_UNITS)
        scorePerUnit[29] = 1

        # Get a score if we added a unit of that type
        reward = (deltaUnitCounts * scorePerUnit).sum()

        assert(deltaTime >= 0)

        # Assume the reward happens right before s2.
        # If the build order involved satisfying some implicit constraints or maybe some waiting time
        # then the reward will be rewarded right at the end.
        # This makes it beneficial for the agent to learn to handle implicit dependencies by itself, but it can still fall back on them without too big of a loss.
        reward *= math.pow(GAMMA_PER_SECOND, deltaTime)
        return reward

    def determine_action(self, s1, s2, unit):
        pass

    def loadSession(self, s):
        data = json.loads(s)
        if "actions" not in data:
            print("Missing actions")
            return

        states = data["states"]
        actions = data["actions"]
        assert(len(actions) == len(states) - 1)

        goalTensor = self.createGoalTensor(data["goal"])
        self.goalPool.append(goalTensor)

        # Ensure the pool doesn't grow too large
        if len(self.goalPool) > 10000:
            self.goalPool[random.randint(0, len(self.goalPool)-1)] = self.goalPool[-1]
            self.goalPool.pop()

        goals = random.choices(self.goalPool, k=min(len(self.goalPool), 4))
        goals.append(goalTensor)

        tensor_states_pre = [self.createState(s) for s in states]

        for goal in goals:
            total_reward = 0
            tensor_states = [self.combineStateAndGoal(t, goal) for t in tensor_states_pre]

            for i in range(len(tensor_states)-1):
                s1 = states[i]
                s2 = states[i+1]
                a1 = unitIndexMap[actions[i]]
                t1 = tensor_states[i]
                t2 = tensor_states[i+1]

                if not data["failed"]:
                    if np.all((t1.numpy() - t2.numpy()) == 0):
                        print(t1)
                        print(t2)
                        print(s1)
                        print(s2)
                        exit(1)

                a2 = unitIndexMap[actions[i+1]] if i+1 < len(actions) else None
                deltaTime = s2["time"] - s1["time"]

                reward = self.calculate_reward(t1, t2, goalTensor, deltaTime)
                terminal_state = a2 is None

                total_reward += reward
                assert(deltaTime >= 0)

                # Skip terminal states at the moment
                if terminal_state:
                    continue

                transition = Transition(state=t1, action=a1, next_state=t2, reward=reward, next_action=a2, deltaTime=deltaTime)
                if random.uniform(0,1) < 0.1:
                    test_data.append(transition)
                else:
                    self.push(transition)

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
        layers.append(nn.BatchNorm1d(100))
        layers.append(nn.Linear(100, 100))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(100))
        layers.append(nn.Linear(100, 100))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(100))
        layers.append(nn.Linear(100, 100))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(100))
        layers.append(nn.Linear(100, 100))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.Linear(100, 100))
        # layers.append(nn.Linear(100, 100))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(100))
        layers.append(nn.Linear(100, NUM_ACTIONS))
        self.seq = nn.Sequential(*layers)

    def forward(self, inputTensor):
        x = self.seq(inputTensor)
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
    fs = fs[:100]
    # random.shuffle(fs)
    for i in range(len(fs)):
        print(f"{i}/{len(fs)}")
        p = fs[i]
        f = open(data_path + "/" + p)
        s = f.read()
        f.close()
        memory.loadSession(s)
        if optimization_steps_per_load > 0:
            optimize(optimization_steps_per_load)
    print("Done")


data_path = "training_data/buildorders/1"
BATCH_SIZE = 512
# GAMMA_PER_SECOND = 0.995
GAMMA_PER_SECOND = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 4000
TEMPERATURE_START = 10
TEMPERATURE_END = 1
TEMPERATURE_DECAY = 8000
TARGET_UPDATE = 200

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.00001)
memory = ReplayMemory(100000)
test_data = []
qvalue_range = []


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
test_losses = []
episode_index = 0

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


def evaluate_batch(transitions):
    batch_size = len(transitions)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)

    # No non-final states, will cause some torch errors
    any_non_final = len([s for s in batch.next_state if s is not None]) > 0
    if any_non_final > 0:
        non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state if s is not None])

    state_batch = torch.cat([s.unsqueeze(0) for s in batch.state])
    assert state_batch.size() == (batch_size, TENSOR_INPUT_SIZE), (state_batch.size(), (batch_size, TENSOR_INPUT_SIZE))

    action_batch = torch.tensor(batch.action).unsqueeze(1)
    assert(action_batch.size() == (batch_size,1))
    reward_batch = torch.tensor(batch.reward, dtype=torch.float)
    assert reward_batch.size() == (batch_size,)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    assert state_action_values.size() == (batch_size, 1)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(batch_size, device=device)
    if any_non_final:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    assert next_state_values.size() == (batch_size,)

    # torch.set_printoptions(threshold=10000)

    # Compute the expected Q values
    gammas = np.power(GAMMA_PER_SECOND, batch.deltaTime)
    expected_state_action_values = (next_state_values * torch.tensor(gammas, dtype=torch.float)) + reward_batch
    assert expected_state_action_values.size() == (batch_size,)

    # Compute Huber loss
    mse = nn.MSELoss(reduction='none')
    transition_losses = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
    # transition_losses = mse(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = transition_losses.mean()
    memory.insert(transitions, transition_losses)
    return state_action_values, expected_state_action_values.unsqueeze(1), loss

def optimize_model():
    global episode_index
    episode_index += 1

    if len(memory) < BATCH_SIZE:
        return

    policy_net.train()
    transitions = memory.sample(BATCH_SIZE)
    calculated_values, expected_values, loss = evaluate_batch(transitions)

    qvalue_range.append((episode_index, expected_values.numpy().mean(), expected_values.numpy().std()))
    
    print(loss.item(), np.mean(expected_values.numpy()))
    losses.append([episode_index, loss.item()])

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def evaluate_test_loss():
    policy_net.eval()
    transitions = test_data
    calculated_values, expected_values, loss = evaluate_batch(transitions)
    test_losses.append([episode_index, loss.item()])


plt.ioff()
episode = 0

epss = []
temps = []

def plot_loss():
    if len(test_losses) == 0:
        return

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(2,3,1)
    plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    ax.set_yscale('log')
    losses_n = np.array(losses)
    plt.plot(losses_n[:,0], losses_n[:,1])
    losses_n = np.array(test_losses)
    plt.plot(losses_n[:,0], losses_n[:,1])

    ax = fig.add_subplot(2,3,2)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Q value')
    qvalue_range_n = np.array(qvalue_range)
    plt.plot(qvalue_range_n[:,0], qvalue_range_n[:,1] - qvalue_range_n[:,2])
    plt.plot(qvalue_range_n[:,0], qvalue_range_n[:,1] + qvalue_range_n[:,2])

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



def optimize_epoch():
    optimize(int(len(memory) / BATCH_SIZE))

def optimize(steps: int):
    global episode
    for i in range(steps):
        optimize_model()
        episode += 1
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    evaluate_test_loss()
    plot_loss()
    # plt.show()


if __name__ == "__main__":
    load_all(0)
    while True:
        optimize_epoch()
else:
    load_all(200)
    pass
