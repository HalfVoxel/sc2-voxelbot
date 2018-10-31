import json
import os

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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TERRAN_REAPER = 49


def dist(p1, p2):
    return math.sqrt(math.pow(p2["x"] - p1["x"], 2) + math.pow(p2["y"] - p1["y"], 2))


def unitDistance(unit1, unit2):
    return dist(unit1["position"], unit2["position"])


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def createState(self, unit, state):
        nearby = []
        for unit2 in state["units"]:
            if unit != unit2 and unitDistance(unit, unit2) < distanceThreshold:
                nearby.append(unit2)

        sort(nearby, key=lambda u: unitDistance(unit, u))

        originx = unit["position"]["x"]
        originy = unit["position"]["y"]

        maxAllies = 8
        maxEnemies = 8
        allyNearby = []
        enemyNearby = []

        selfUnit = [
            0,  # Total allies
            0,  # Total enemies
            unit["energy"],
            unit["is_flying"],
            unit["is_burrowed"],
            unit["is_powered"],
            unit["radius"],
            unit["detect_range"],
            unit["weapon_cooldown"],
            unit["build_progress"],
            unit["shield"],
            unit["health"],
            unit["health"]/math.max(1, unit["health_max"]),
        ]

        for u in nearby:
            relativeUnit = [
                1,  # Does unit exist
                u["position"]["x"] - originx,
                u["position"]["y"] - originy,
                unitDistance(unit, u),
                u["energy"],
                u["is_flying"],
                u["is_burrowed"],
                u["is_powered"],
                u["radius"],
                u["detect_range"],
                u["weapon_cooldown"],
                u["build_progress"],
                u["shield"],
                u["health"],
                u["health"]/math.max(1, u["health_max"]),
                # In attack range?
            ]
            if u["owner"] == state["playerID"]:
                if len(allyNearby) < maxAllies:
                    allyNearby.append(relativeUnit)
                selfUnit[0] += 1
            else:
                if len(enemyNearby) < maxEnemies:
                    enemyNearby.append(relativeUnit)
                selfUnit[1] += 1

        dummyUnit = [
            0,  # Does unit exist
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]

        while len(allyNearby) < maxAllies:
            allyNearby.append(dummyUnit)
        while len(enemyNearby) < maxEnemies:
            enemyNearby.append(dummyUnit)

        enemyTensor = torch.tensor(enemyNearby, dtype=torch.float)
        allyTensor = torch.tensor(allyNearby, dtype=torch.float)
        selfTensor = torch.tensor(selfUnit, dtype=torch.float)
        return [selfTensor, allyTensor, enemyTensor]

        # Input:
        # 2 x 8 x [
        # SerializedPos position; -> relative pos
        # UNIT_TYPEID unit_type;
        # UNIT_TYPEID canonical_unit_type;
        # Unit::DisplayType display_type;
        # Unit::CloakState cloak;
        # int tag;
        # int owner; -> NOP
        # float energy;
        # float energy_max;
        # bool is_flying;
        # bool is_burrowed;
        # bool is_powered;
        # float radius;
        # float facing;
        # float detect_range;
        # float weapon_cooldown;
        # float build_progress;
        # float shield;
        # float health;
        # float health_max; -> health fraction
        # ] 

    def calculate_reward(self, s1, s2, unit):
        tag2unit = {}
        for unit2 in s2["units"]:
            tag2unit[unit2["tag"]] = unit2

        # Will contain all (unit, unit) pairs such that
        # they point to the same unit, the first elemeny is in state1 and the second one is in state2
        # If the unit does not exist in state2 then the second element will be None.
        # If the unit does not exist in state1 then the unit will be ignored

        nextUnitState = []
        reward = 0

        for unit2 in s1["units"]:
            if unit2["tag"] in tag2unit:
                nextUnitState.append((unit2, tag2unit(unit2["tag"])))
            else:
                nextUnitState.append((unit2, None))

        for (u1, u2) in nextUnitState:
            if u1["owner"] == s1["playerID"]:
                # Ally unit
                if u2 is None:
                    # Lost a unit
                    if u1 == unit:
                        # This unit died!!
                        reward -= 100
                    else:
                        reward -= 30
                else:
                    shieldDiff = u2["shield"] - u1["shield"]
                    healthDiff = u2["health"] - u1["health"]
                    # Note: weigh health higher than shields
                    totalDiff = 0.5 * shieldDiff + healthDiff
                    if u1 == unit:
                        if totalDiff < 0:
                            # This unit lost health
                            reward -= 10
                        elif totalDiff > 0:
                            reward += 1
                    else:
                        if totalDiff < 0:
                            # An ally unit lost health
                            reward -= 10
                        elif totalDiff > 0:
                            reward += 1
            else:
                # Enemy unit
                if u2 is None:
                    # Killed a unit!
                    reward += 30
                else:
                    shieldDiff = u2["shield"] - u1["shield"]
                    healthDiff = u2["health"] - u1["health"]
                    # Note: weigh health higher than shields
                    totalDiff = 0.5 * shieldDiff + healthDiff
                    if totalDiff < 0:
                        # An enemy unit lost health
                        reward += 5
                    elif totalDiff > 0:
                        reward -= 1

        terminal_state = unit["tag"] not in tag2unit
        return reward, terminal_state

    def determine_action(self, s1, s2, unit):
        pass

    def loadSession(self, s):
        data = json.loads(s)
        states = data["states"]
        for i in range(len(states)):
            s1 = states[i]
            s2 = states[i+1] if i + 1 < len(states) else None
            if s2 == None:
                continue
            distanceThreshold = 10
            for unit in s1["units"]:
                if unit["unit_type"] == TERRAN_REAPER and unit["owner"] == s1["playerID"]:
                    # Got a unit that we want to add a sample for
                    t1 = self.createState(unit, s1)
                    t2 = self.createState(unit, s2)
                    reward, terminal_state = self.calculate_reward(s1, s2, unit)
                    if terminal_state:
                        t2 = None
                    action = unit["action"]
                    self.push(Transition(t1, action, t2, reward))

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.lin1_1 = nn.Linear(TENSOR_ALLY_SIZE1, 32)
        self.lin1_2 = nn.Linear(TENSOR_ENEMY_SIZE1, 32)
        self.lin1_3 = nn.Linear(TENSOR_SELF_SIZE, 32)

        self.lin2_1 = nn.Linear(32, 20)
        self.lin2_2 = nn.Linear(32, 20)
        self.lin2_3 = nn.Linear(32, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(20)

        self.drop1 = nn.Dropout(0.5)

        self.lin3 = nn.Linear(TENSOR_SELF_SIZE + TENSOR_ALLY_SIZE1 + TENSOR_ENEMY_SIZE1, 20)
        self.bn4 = nn.BatchNorm1d(20)
        self.lin4 = nn.Linear(20, NUM_ACTIONS)

    def forward(self, selfTensor, allyTensor, enemyTensor):
        # selfTensor: B x 13
        # allyTensor: B x 8 x 15
        # enemyTensor: B x 8 x 15
        x = F.relu(self.lin1_1x(selfTensor))
        x = F.relu(self.bn1(self.lin2_1(selfTensor)))
        selfTens = self.drop1(x)

        x = F.relu(self.lin1_2(allyTensor))
        x = F.relu(self.bn2(self.lin2_2(x)))
        allyTens = self.drop1(x)

        x = F.relu(self.lin1_3(enemyTensor))
        x = F.relu(self.bn3(self.lin2_3(x)))
        enemyTens = self.drop1(x)

        allyTens = allyTens.sum(dim=1)
        enemyTens = enemyTens.sum(dim=1)

        allTens = torch.cat((selfTens, allyTens, enemyTens), dim=1)
        x = F.relu(self.bn4(self.lin3(allTens)))
        x = self.lin4(x)

        return x  # B x NUM_ACTIONS


def predict(s, unitTag):
    state = json.loads(s)
    print()
    t1 = memory.createState(unit, state)
    return select_action(t1)


def addSession(s):
    path = "training_data/1"
    index = len(os.listdir(path))
    with open(path + "/session" + str(index) + ".json", "w") as f:
        f.write(s)

    loadSession(s)
    memory.loadSession(s)
    pass

BATCH_SIZE = 128
GAMMA_PER_SECOND = 0.98
TICKS_PER_STATE = 25
TICKS_PER_SECOND = 22.4
GAMMA = math.pow(GAMMA_PER_TICK, TICKS_PER_STATE / TICKS_PER_SECOND)
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_ACTIONS = 9
TENSOR_SELF_SIZE = 13
TENSOR_ALLY_SIZE0 = 8
TENSOR_ALLY_SIZE1 = 15
TENSOR_ENEMY_SIZE0 = 8
TENSOR_ENEMY_SIZE1 = 15

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state[0], state[1], state[2]).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)


episode_durations = []


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
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states0 = torch.cat([s[0] for s in batch.next_state if s is not None])
    non_final_next_states1 = torch.cat([s[1] for s in batch.next_state if s is not None])
    non_final_next_states2 = torch.cat([s[2] for s in batch.next_state if s is not None])
    state_batch0 = torch.cat([s[0] for s in batch.state]) # selfTensor
    state_batch1 = torch.cat([s[1] for s in batch.state]) # allyTensor
    state_batch2 = torch.cat([s[1] for s in batch.state]) # enemyTensor
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch0, state_batch1, state_batch2).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states0, non_final_next_states1, non_final_next_states2).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def optimize():
    optimize_model()

# num_episodes = 50
# for i_episode in range(num_episodes):
#     # Initialize the environment and state
#     env.reset()
#     last_screen = get_screen()
#     current_screen = get_screen()
#     state = current_screen - last_screen
#     for t in count():
#         # Select and perform an action
#         action = select_action(state)
#         _, reward, done, _ = env.step(action.item())
#         reward = torch.tensor([reward], device=device)

#         # Observe new state
#         last_screen = current_screen
#         current_screen = get_screen()
#         if not done:
#             next_state = current_screen - last_screen
#         else:
#             next_state = None

#         # Store the transition in memory
#         memory.push(state, action, next_state, reward)

#         # Move to the next state
#         state = next_state

#         # Perform one step of the optimization (on the target network)
#         optimize_model()
#         if done:
#             episode_durations.append(t + 1)
#             plot_durations()
#             break
#     # Update the target network
#     if i_episode % TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())

# print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.show()

