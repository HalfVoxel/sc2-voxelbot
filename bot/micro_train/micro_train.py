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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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

    def push(self, transition: Transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def createState(self, unit, state):
        nearby = []
        for unit2 in state["units"]:
            if unit != unit2 and unitDistance(unit, unit2) < NEARBY_UNIT_DISTANCE_THRESHOLD:
                nearby.append(unit2)

        nearby.sort(key=lambda u: unitDistance(unit, u))

        originx = unit["position"]["x"]
        originy = unit["position"]["y"]

        maxAllies = TENSOR_ALLY_SIZE0
        maxEnemies = TENSOR_ENEMY_SIZE0
        allyNearby = []
        enemyNearby = []

        selfUnit = [
            0,  # Total allies
            0,  # Total enemies
            # unit["energy"],
            # unit["is_flying"],
            # unit["is_burrowed"],
            # unit["is_powered"],
            # unit["radius"],
            # unit["detect_range"],
            1 if unit["weapon_cooldown"] > 0.1 else 0,
            # unit["build_progress"],
            unit["shield"] / 100.0,  # Make values be roughly 1 in most cases
            unit["health"] / 100.0,
            unit["health"]/max(1, unit["health_max"]),
        ]

        for u in nearby:
            dx = u["position"]["x"] - originx
            dy = u["position"]["y"] - originy
            angle = math.atan2(dy, dx)
            relativeUnit = [
                1,  # Does unit exist
                dx,
                dy,
                angle,
                unitDistance(unit, u),
                1 if unitDistance(unit, u) <= 5 else 0,  # In range
                # u["energy"],
                # u["is_flying"],
                # u["is_burrowed"],
                # u["is_powered"],
                # u["radius"],
                # u["detect_range"],
                # u["weapon_cooldown"],
                # u["build_progress"],
                u["shield"] / 100.0,  # Make values be roughly 1 in most cases
                u["health"] / 100.0,
                u["health"]/max(1, u["health_max"]),
                # In attack range?
            ]
            assert(len(relativeUnit) == TENSOR_ALLY_SIZE1)
            assert(len(relativeUnit) == TENSOR_ENEMY_SIZE1)
            if u["owner"] == state["playerID"]:
                if len(allyNearby) < maxAllies:
                    allyNearby.append(relativeUnit)
                selfUnit[0] += 1
            else:
                if len(enemyNearby) < maxEnemies:
                    enemyNearby.append(relativeUnit)
                selfUnit[1] += 1

        print(f"Nearby units: {len(nearby)}, of which enemies: {len(enemyNearby)}")

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
        ]

        assert(len(dummyUnit) == TENSOR_ALLY_SIZE1)
        assert(len(dummyUnit) == TENSOR_ENEMY_SIZE1)

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
                nextUnitState.append((unit2, tag2unit[unit2["tag"]]))
            else:
                nextUnitState.append((unit2, None))

        distanceToEnemy = 10000
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
                            reward += 0.1
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
                    # distanceToEnemy = min(distanceToEnemy, unitDistance(unit, u2))
                    distanceToEnemy = min(distanceToEnemy, unitDistance(unit, u1))
                    shieldDiff = u2["shield"] - u1["shield"]
                    healthDiff = u2["health"] - u1["health"]
                    # Note: weigh health higher than shields
                    totalDiff = 0.5 * shieldDiff + healthDiff
                    if totalDiff < 0:
                        # An enemy unit lost health
                        reward += 5
                    elif totalDiff > 0:
                        reward -= 1

        # Avoid hiding in a corner
        reward -= 0.1
        if distanceToEnemy > 8:
            reward -= 1

        terminal_state = unit["tag"] not in tag2unit

        if terminal_state and False:
            reward = 0
        else:
            reward = 0
            # reward = 1 if tag2unit[unit["tag"]]["action"] == 0 else 0
            if unit["action"] == 0:
                reward += 1 if distanceToEnemy < 5 else 0
            if unit["action"] == 1:
                reward += 1 if distanceToEnemy > 5 else 0

        print(f"Reward {reward}")

        reward = 1 if unit["weapon_cooldown"] > 0.1 else 0

        return reward, terminal_state

    def determine_action(self, s1, s2, unit):
        pass

    def loadSession(self, s):
        data = json.loads(s)
        # print(s)
        states = data["states"]
        for i in range(len(states)):
            s1 = states[i]
            s2 = states[i+1] if i + 1 < len(states) else None
            if s2 == None:
                continue
            for unit in s1["units"]:
                if unit["unit_type"] == TERRAN_REAPER and unit["owner"] == s1["playerID"]:
                    # Got a unit that we want to add a sample for
                    t1 = self.createState(unit, s1)
                    t2 = self.createState(unit, s2)
                    reward, terminal_state = self.calculate_reward(s1, s2, unit)
                    if terminal_state:
                        t2 = None
                    action = unit["action"]
                    self.push(Transition(state=t1, action=action, next_state=t2, reward=reward))

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        # self.lin1_1 = nn.Linear(TENSOR_SELF_SIZE, 32)
        # self.lin1_2 = nn.Linear(TENSOR_ALLY_SIZE1, 32)
        # self.lin1_3 = nn.Linear(TENSOR_ENEMY_SIZE1, 32)

        # self.lin2_1 = nn.Linear(32, 20)
        # self.lin2_2 = nn.Linear(32, 20)
        # self.lin2_3 = nn.Linear(32, 20)
        # # self.bn1 = nn.BatchNorm1d(20)
        # # self.bn2 = nn.BatchNorm1d(20)
        # # self.bn3 = nn.BatchNorm1d(20)

        # self.drop1 = nn.Dropout(0.5)

        # self.lin3 = nn.Linear(20 + 20 + 20, 20)
        # # self.bn4 = nn.BatchNorm1d(20)
        # self.lin4 = nn.Linear(20, NUM_ACTIONS)
        self.lin5 = nn.Linear(TENSOR_SELF_SIZE, 20)
        self.lin6 = nn.Linear(20, NUM_ACTIONS)

    def forward(self, selfTensor, allyTensor, enemyTensor):
        x = F.sigmoid(self.lin5(selfTensor))
        x = F.sigmoid(self.lin6(x))
        return x
        # x = self.lin2_1(x)
        # x = F.relu(x)
        # # selfTens = self.drop1(x)

        # return self.lin5(x)

        # selfTensor: B x 13
        # allyTensor: B x 8 x 15
        # enemyTensor: B x 8 x 15
        x = F.relu(self.lin1_1(selfTensor))
        x = self.lin2_1(x)
        x = F.relu(x)
        selfTens = self.drop1(x)

        x = F.relu(self.lin1_2(allyTensor))
        x = self.lin2_2(x)
        # Flatten all nearby units as batches for batch normalization to work
        origShape = x.size()
        x = x.view(-1, x.size()[-1])
        x = F.relu(x)
        x = x.view(*origShape)
        allyTens = self.drop1(x)

        x = F.relu(self.lin1_3(enemyTensor))
        x = self.lin2_3(x)
        # Flatten all nearby units as batches for batch normalization to work
        origShape = x.size()
        x = x.view(-1, x.size()[-1])
        x = F.relu(x)
        x = x.view(*origShape)
        enemyTens = self.drop1(x)

        allyTens = allyTens.sum(dim=1)
        enemyTens = enemyTens.sum(dim=1)

        allTens = torch.cat((selfTens, allyTens, enemyTens), dim=1)
        # x = F.relu(self.bn4(self.lin3(allTens)))
        x = F.relu(self.lin3(allTens))
        x = self.lin4(x)

        print("OUTPUT: " + str(x.size()))
        return x  # B x NUM_ACTIONS


def predict(s, unitTag):
    state = json.loads(s)
    #print(state)
    unit = None

    for u in state["units"]:
        if u["tag"] == unitTag:
            unit = u
            break

    assert unit is not None, "unit did not exist in state"
    t1 = memory.createState(unit, state)
    print(t1)
    return select_action(t1)


def addSession(s):
    index = len(os.listdir(data_path))
    with open(data_path + "/session" + str(index) + ".json", "w") as f:
        f.write(s)

    memory.loadSession(s)
    pass


def load_all(optimization_steps_per_load: int):
    print("Loading training data...")
    for p in os.listdir(data_path):
        f = open(data_path + "/" + p)
        s = f.read()
        f.close()
        memory.loadSession(s)
        optimize(optimization_steps_per_load)
    print("Done")


data_path = "training_data/1"
BATCH_SIZE = 64
GAMMA_PER_SECOND = 0 # 0.94
TICKS_PER_STATE = 12
TICKS_PER_SECOND = 22.4
GAMMA = math.pow(GAMMA_PER_SECOND, TICKS_PER_STATE / TICKS_PER_SECOND)
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 100
NUM_ACTIONS = 12
TENSOR_SELF_SIZE = 6 # 13
TENSOR_ALLY_SIZE0 = 1
TENSOR_ALLY_SIZE1 = 9 # 15
TENSOR_ENEMY_SIZE0 = 1
TENSOR_ENEMY_SIZE1 = 9 # 15
NEARBY_UNIT_DISTANCE_THRESHOLD = 150

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
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
            policy_net.eval()
            q = policy_net(state[0].unsqueeze(0), state[1].unsqueeze(0), state[2].unsqueeze(0))
            mx = q.max(1)
            print(f"Expected reward: {mx[0].item()}")
            res = mx[1].item()
            policy_net.train()
            return res
    else:
        return torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)


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
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)

    # No non-final states, will cause some torch errors
    any_non_final = len([s for s in batch.next_state if s is not None]) > 0
    if any_non_final > 0:
        non_final_next_states0 = torch.cat([s[0].unsqueeze(0) for s in batch.next_state if s is not None])
        non_final_next_states1 = torch.cat([s[1].unsqueeze(0) for s in batch.next_state if s is not None])
        non_final_next_states2 = torch.cat([s[2].unsqueeze(0) for s in batch.next_state if s is not None])

    state_batch0 = torch.cat([s[0].unsqueeze(0) for s in batch.state])  # selfTensor
    state_batch1 = torch.cat([s[1].unsqueeze(0) for s in batch.state])  # allyTensor
    state_batch2 = torch.cat([s[2].unsqueeze(0) for s in batch.state])  # enemyTensor
    action_batch = torch.tensor(batch.action).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float)
    k = state_batch0[:,2]
    print("REWARDS", reward_batch, k)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch0, state_batch1, state_batch2).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if any_non_final:
        next_state_values[non_final_mask] = target_net(non_final_next_states0, non_final_next_states1, non_final_next_states2).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    print("BLAH", expected_state_action_values.unsqueeze(1).size(), state_action_values.size())
    print(state_action_values)
    print(reward_batch)
    print(torch.abs(expected_state_action_values.unsqueeze(1) - state_action_values))
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    print(loss.item())
    losses.append(loss.item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in policy_net.parameters():
        # param.grad.data.clamp_(-1, 1)
    optimizer.step()


plt.ioff()
episode = 0


def plot_loss():
    plt.clf()
    durations_t = torch.tensor(losses, dtype=torch.float)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    ax.set_yscale('log')
    plt.plot(durations_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize(steps: int):
    global episode
    for i in range(steps):
        optimize_model()
        episode += 1
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    plot_loss()
    # plt.show()



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
    load_all(8)
    while True:
        optimize(10)
else:
    load_all(8)
