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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'next_action', 'reward'))

TERRAN_REAPER = 49


def dist(p1, p2):
    return math.sqrt(math.pow(p2["x"] - p1["x"], 2) + math.pow(p2["y"] - p1["y"], 2))


def unitDistance(unit1, unit2):
    return dist(unit1["position"], unit2["position"])


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        # self.memory = []
        # self.position = 0
        self.durations = []
        self.health_diffs = []
        self.total_rewards = []
        self.all_actions = []
        self.max_error = 1000
        self.error_buckets = []
        for i in range(10):
            self.error_buckets.append([])

        self.count = 0

    def push(self, transition: Transition):
        """Saves a transition."""
        # if len(self.memory) < self.capacity:
            # self.memory.append(None)
        # self.memory[self.position] = transition
        # self.position = (self.position + 1) % self.capacity
        random.choice(self.error_buckets).append(transition)
        self.count += 1

    def sample(self, batch_size):
        result = []
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
        #return random.sample(self.memory, batch_size)

    def discard_random(self):
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
        assert len(samples) == len(errors)
        for i in range(len(samples)):
            # Discard a random sample
            while self.count >= self.capacity:
                self.discard_random()

            bucket_idx = max(0, min(len(self.error_buckets)-1, math.floor(len(self.error_buckets) * errors[i].item() / self.max_error)))
            self.error_buckets[bucket_idx].append(samples[i])
            self.count += 1

    def createState(self, unit, state, last_attacked_at):
        nearby = []
        for unit2 in state["units"]:
            if unit != unit2 and unitDistance(unit, unit2) < (NEARBY_UNIT_DISTANCE_THRESHOLD if unit2["owner"] != unit["owner"] else NEARBY_ALLY_UNIT_DISTANCE_THRESHOLD):
                nearby.append(unit2)

        nearby.sort(key=lambda u: unitDistance(unit, u))

        originx = unit["position"]["x"]
        originy = unit["position"]["y"]

        maxAllies = TENSOR_ALLY_SIZE0
        maxEnemies = TENSOR_ENEMY_SIZE0
        allyNearby = []
        enemyNearby = []

        lastAttackedTick = last_attacked_at[unit["tag"]] if unit["tag"] in last_attacked_at else -1000
        maxSeconds = 50
        secondsSinceAttacked = min(maxSeconds, (state["tick"] - lastAttackedTick)/TICKS_PER_SECOND)
        selfUnit = [
            0,  # Total allies
            0,  # Total enemies
            # originx,
            # originy,
            # unit["energy"],
            # unit["is_flying"],
            # unit["is_burrowed"],
            # unit["is_powered"],
            # unit["radius"],
            # unit["detect_range"],
            0 if unit["weapon_cooldown"] == 0 else 1,
            # unit["build_progress"],
            # (unit["health"] + unit["shield"]) / 100.0,  # Make values be roughly 1 in most cases
            unit["health"]/max(1, unit["health_max"]),  # TODO: shield
            secondsSinceAttacked,
        ]

        closestEnemy = None
        for u in nearby:
            dx = u["position"]["x"] - originx
            dy = u["position"]["y"] - originy
            angle = math.atan2(dy, dx)
            relativeUnit = [
                1,  # Does unit exist
                # dx,
                # dy,
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
                (u["health"] + u["shield"]) / (u["health_max"] + u["shield_max"]),  # Make values be roughly 1 in most cases
                # u["health"] / 100.0,
                # u["health"]/max(1, u["health_max"]),
                # In attack range?
            ]
            # assert(len(relativeUnit) == TENSOR_ALLY_SIZE1)
            # assert(len(relativeUnit) == TENSOR_ENEMY_SIZE1)
            if u["owner"] == state["playerID"]:
                if len(allyNearby) < maxAllies:
                    allyNearby.append(relativeUnit)
                selfUnit[0] += 1
            else:
                if closestEnemy is None:
                    closestEnemy = u

                if len(enemyNearby) < maxEnemies:
                    enemyNearby.append(relativeUnit)
                selfUnit[1] += 1

        # print(f"Nearby units: {len(nearby)}, of which enemies: {len(enemyNearby)}")

        dummyUnit = [
            0,  # Does unit exist
            0,
            0,
            0,
            0,
        ]

        assert(len(selfUnit) == TENSOR_SELF_SIZE)
        assert(len(dummyUnit) == TENSOR_ALLY_SIZE1)
        assert(len(dummyUnit) == TENSOR_ENEMY_SIZE1)

        while len(allyNearby) < maxAllies:
            allyNearby.append(dummyUnit)
        while len(enemyNearby) < maxEnemies:
            enemyNearby.append(dummyUnit)

        enemyTensor = torch.tensor(enemyNearby, dtype=torch.float)
        allyTensor = torch.tensor(allyNearby, dtype=torch.float)
        selfTensor = torch.tensor(selfUnit, dtype=torch.float)

        # enemyTensor = torch.zeros((1, 2))
        # allyTensor = torch.zeros((1, 1))

        # healthIndex = int(min(3, round(3*unit["health"]/max(1, unit["health_max"]))))
        # nearbyEnemyIndex = len(enemyNearby)
        # enemyHealthIndex = min(3, round(3*((closestEnemy["health"] + closestEnemy["shield"])/(closestEnemy["health_max"] + closestEnemy["shield_max"])))) if closestEnemy is not None else 0
        # enemyDistIndex = unitDistance(closestEnemy, unit) if closestEnemy is not None else 0
        # enemyDistIndex = min(5, int(math.floor(enemyDistIndex)))

        # index = 0
        # m = 1

        # index += m * healthIndex
        # assert healthIndex <= 4
        # m *= 3+1

        # index += m * (1 if unit["weapon_cooldown"] == 0 else 0)
        # m *= 2

        # index += m * nearbyEnemyIndex
        # assert nearbyEnemyIndex <= maxEnemies
        # m *= maxEnemies+1

        # index += m * int(enemyHealthIndex)
        # assert int(enemyHealthIndex) <= 4
        # m *= 3+1

        # index += m * enemyDistIndex
        # assert enemyDistIndex <= 5
        # m *= 5+1

        # assert(m == TABLE_SIZE)
        # assert(index < TABLE_SIZE)

        # dat = torch.tensor(index, dtype=torch.long)
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

        allyUnits = []
        for u in s1["units"]:
            if u["owner"] == unit["owner"]:
                allyUnits.append(u)

        distanceToEnemy = 10000
        for (u1, u2) in nextUnitState:
            currentReward = 0
            if u1["owner"] == s1["playerID"]:
                # Ally unit
                if u2 is None:
                    # Lost a unit
                    if u1 == unit:
                        # This unit died!!
                        currentReward -= 1000
                    else:
                        currentReward -= 30
                else:
                    shieldDiff = u2["shield"] - u1["shield"]
                    healthDiff = u2["health"] - u1["health"]
                    # Note: weigh health higher than shields
                    totalDiff = 0.5 * shieldDiff + healthDiff
                    if u1 == unit:
                        if totalDiff < 0:
                            # This unit lost health
                            currentReward -= healthDiff
                        elif totalDiff > 0:
                            currentReward += 0.1
                    else:
                        if totalDiff < 0:
                            # An ally unit lost health
                            currentReward -= healthDiff * 0.5
                        elif totalDiff > 0:
                            currentReward += 0.1
            else:
                # Enemy unit
                if u2 is None:
                    # Killed a unit!
                    currentReward += 100
                else:
                    # distanceToEnemy = min(distanceToEnemy, unitDistance(unit, u2))
                    distanceToEnemy = min(distanceToEnemy, unitDistance(unit, u1))
                    shieldDiff = u2["shield"] - u1["shield"]
                    healthDiff = u2["health"] - u1["health"]
                    # Note: weigh health higher than shields
                    totalDiff = 0.5 * shieldDiff + healthDiff
                    if totalDiff < 0:
                        # An enemy unit lost health
                        currentReward += healthDiff * 0.5
                    elif totalDiff > 0:
                        currentReward -= 1

            if currentReward != 0:
                # Split up the reward among the nearby units
                creditWeight = 0
                totalCreditWeight = 0
                for u in allyUnits:
                    d = unitDistance(u, u1)
                    w = max(0, 1 - d/10.0)
                    totalCreditWeight += w
                    if u == unit:
                        creditWeight += w

                normalizedCreditWeight = creditWeight / totalCreditWeight if creditWeight > 0 else 0
                reward += currentReward * normalizedCreditWeight

        # Avoid hiding in a corner
        if distanceToEnemy >= 8:
            reward -= 0.3
        else:
            reward -= 0.3 * (distanceToEnemy/8)

        terminal_state = unit["tag"] not in tag2unit

        # if terminal_state and False:
        #     reward = 0
        # else:
        #     reward = 0
        #     # reward = 1 if tag2unit[unit["tag"]]["action"] == 0 else 0
        #     if unit["action"] == 0:
        #         reward += unit["weapon_cooldown"]
        #     if unit["action"] == 1:
        #         reward += unit["weapon_cooldown"]

        # print(f"Reward {reward}")

        # reward = 1 if unit["weapon_cooldown"] > 0.1 else 0

        return reward, terminal_state

    def determine_action(self, s1, s2, unit):
        pass

    def loadSession(self, s):
        data = json.loads(s)
        # print(s)
        states = data["states"]
        total_reward = 0
        last_attacked_at = {}
        last_health = {}

        tensor_states = []
        for i in range(len(states)):
            s1 = states[i]
            statesForUnit = {}
            for unit in s1["units"]:
                hp = unit["health"] + unit["shield"]
                if hp < last_health[unit["tag"]]:
                    last_attacked_at[unit["tag"]] = s1["tick"]
                last_health[unit["tag"]] = hp

                # unit["weapon_cooldown"] = 1 if random.uniform(0, 1) < 0.5 else 0
                if unit["unit_type"] == TERRAN_REAPER and unit["owner"] == s1["playerID"]:
                    # Got a unit that we want to add a sample for
                    t1 = self.createState(unit, s1, last_attacked_at)
                    statesForUnit[unit["tag"]] = (unit, t1)
                    self.all_actions.append(unit["action"])

            tensor_states.append(statesForUnit)

        for i in range(len(tensor_states)-1):
            s1 = states[i]
            s2 = states[i+1]

            for tag, (t1_unit, t1) in tensor_states[i].items():
                if tag in tensor_states[i+1]:
                    t2_unit, t2 = tensor_states[i+1][tag]
                else:
                    t2_unit, t2 = (None, None)

                reward, terminal_state = self.calculate_reward(s1, s2, t1_unit)
                t1_action = t1_unit["action"]

                assert terminal_state == (t2 is None)
                if terminal_state:
                    t2_action = None
                else:
                    t2_action = t2_unit["action"]

                total_reward += reward
                self.push(Transition(state=t1, action=t1_action, next_state=t2, reward=reward, next_action=t2_action))

        health_diff = self.total_health(states[0]) - self.total_health(states[-1])
        self.total_rewards.append(total_reward)
        self.durations.append(len(states))
        self.health_diffs.append(health_diff)

    def total_health(self, state):
        t = 0
        for unit in state["units"]:
            if unit["owner"] != state["playerID"]:
                t += unit["shield"] + unit["health"]

        return t

    def __len__(self):
        return self.count


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.lin1_1 = nn.Linear(TENSOR_SELF_SIZE, 20)
        self.lin1_2 = nn.Linear(TENSOR_ALLY_SIZE1, 20)
        self.lin1_3 = nn.Linear(TENSOR_ENEMY_SIZE1, 20)
        # self.enemy_emb = nn.Embedding(NUM_ENEMY_TYPES, 4)

        self.lin2_1 = nn.Linear(20, 20)
        self.lin2_2 = nn.Linear(20, 20)
        self.lin2_3 = nn.Linear(20, 20)
        # self.bn1 = nn.BatchNorm1d(20)
        # self.bn2 = nn.BatchNorm1d(20)
        # self.bn3 = nn.BatchNorm1d(20)

        self.drop1 = nn.Dropout(0.5)

        self.lin3 = nn.Linear(20 + 20 + 20 + 20, 20)
        # self.bn4 = nn.BatchNorm1d(20)
        self.lin4 = nn.Linear(20, NUM_ACTIONS)
        self.lin5 = nn.Linear(TENSOR_SELF_SIZE, 20)
        self.lin6 = nn.Linear(20, NUM_ACTIONS)

        # self.emb = nn.Embedding(TABLE_SIZE, NUM_ACTIONS)

    def forward(self, selfTensor, allyTensor, enemyTensor):
        # print(selfTensor)
        # res = self.emb(selfTensor)
        # print(res)
        # return res

        # x = F.leaky_relu(self.lin5(selfTensor))
        # x = F.leaky_relu(self.lin6(x))
        # return x
        # x = self.lin2_1(x)
        # x = F.leaky_relu(x)
        # selfTens = self.drop1(x)

        # return self.lin5(x)

        # selfTensor: B x 13
        # allyTensor: B x 8 x 15
        # enemyTensor: B x 8 x 15
        x = F.leaky_relu(self.lin1_1(selfTensor))
        # x = self.lin2_1(x)
        # x = F.leaky_relu(x)
        selfTens = self.drop1(x)

        x = F.leaky_relu(self.lin1_2(allyTensor))
        # x = self.lin2_2(x)
        # Flatten all nearby units as batches for batch normalization to work
        # origShape = x.size()
        # x = x.view(-1, x.size()[-1])
        # x = F.leaky_relu(x)
        # x = x.view(*origShape)
        allyTens = self.drop1(x)

        x = F.leaky_relu(self.lin1_3(enemyTensor))
        # y = self.enemy_emb(enemyTypes)
        # x = self.lin2_3(x)
        # Flatten all nearby units as batches for batch normalization to work
        # origShape = x.size()
        # x = x.view(-1, x.size()[-1])
        # x = F.leaky_relu(x)
        # x = x.view(*origShape)
        enemyTens = self.drop1(x)

        allyTens = allyTens.sum(dim=1)
        enemyTens1 = enemyTens.sum(dim=1)
        enemyTens2 = enemyTens.max(dim=1)[0]

        allTens = torch.cat((selfTens, allyTens, enemyTens1, enemyTens2), dim=1)
        # x = F.leaky_relu(self.bn4(self.lin3(allTens)))
        x = F.leaky_relu(self.lin3(allTens))
        x = self.lin4(x)

        # print("OUTPUT: " + str(x.size()))
        return x  # B x NUM_ACTIONS


def predict(s, unitTags, enableExploration):
    state = json.loads(s)
    #print(state)
    unit = None

    result = []
    for tag in unitTags:
        for u in state["units"]:
            if u["tag"] == tag:
                unit = u
                break

        assert unit is not None, "unit did not exist in state"
        t1 = memory.createState(unit, state)
        # print(t1)
        result.append(select_action(t1, enableExploration))

    return result


def addSession(s):
    index = len(os.listdir(data_path))
    with open(data_path + "/session" + str(index) + ".json", "w") as f:
        f.write(s)

    memory.loadSession(s)
    pass


def load_all(optimization_steps_per_load: int):
    print("Loading training data...")
    fs = os.listdir(data_path)
    # random.shuffle(fs)
    for p in fs:
        f = open(data_path + "/" + p)
        s = f.read()
        f.close()
        memory.loadSession(s)
        optimize(optimization_steps_per_load)
    print("Done")


data_path = "training_data/3"
BATCH_SIZE = 128
GAMMA_PER_SECOND = 0.98
TICKS_PER_STATE = 10
TICKS_PER_SECOND = 22.4
GAMMA = math.pow(GAMMA_PER_SECOND, TICKS_PER_STATE / TICKS_PER_SECOND)
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 1
NUM_ACTIONS = 4
TENSOR_SELF_SIZE = 4  # 13
TENSOR_ALLY_SIZE0 = 1
TENSOR_ALLY_SIZE1 = 5  # 15
TENSOR_ENEMY_SIZE0 = 2
TENSOR_ENEMY_SIZE1 = 5  # 15
NEARBY_UNIT_DISTANCE_THRESHOLD = 10
NEARBY_ALLY_UNIT_DISTANCE_THRESHOLD = 50

TABLE_SIZE = 768

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state, enableExploration):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold or not enableExploration:
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
    batch_size = len(transitions)
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
    assert state_batch1.size() == (batch_size,TENSOR_ALLY_SIZE0, TENSOR_ALLY_SIZE1)
    # assert state_batch0.size() == (batch_size,)

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
        next_state_values[non_final_mask] = policy_net(non_final_next_states0, non_final_next_states1, non_final_next_states2).max(1)[0].detach()

    assert next_state_values.size() == (batch_size,)

    # torch.set_printoptions(threshold=10000)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    assert expected_state_action_values.size() == (batch_size,)

    # alpha = 0.01
    # for i in range(batch_size):
    #     v0 = policy_net.emb.weight.data[state_batch0[i]][action_batch[i]]
    #     desired = expected_state_action_values[i]
    #     policy_net.emb.weight.data[state_batch0[i]][action_batch[i]] = v0 * (1-alpha) + desired * alpha

    # print(policy_net.emb.weight.data)

    # print("BLAH", expected_state_action_values.unsqueeze(1).size(), state_action_values.size())
    # print(state_action_values)
    # print(reward_batch)
    # print(torch.abs(expected_state_action_values.unsqueeze(1) - state_action_values))
    # Compute Huber loss
    transition_losses = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
    loss = transition_losses.mean()
    # loss = (state_action_values - expected_state_action_values.unsqueeze(1)).pow(2).sum()
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


    ax = fig.add_subplot(2,3,5)
    counts = [len(bucket) for bucket in memory.error_buckets]
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Memory Bucket')
    ax.set_yscale('log')
    plt.ylabel('Size')

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
    load_all(20)
    while True:
        optimize(200)
else:
    load_all(200)
    pass
