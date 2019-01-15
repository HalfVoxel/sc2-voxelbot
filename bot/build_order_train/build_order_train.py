import json
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import matplotlib
from replay_memory import ReplayMemory
from qlearning import QLearning
from a2c import A2C
from build_order_loader import BuildOrderLoader, TENSOR_INPUT_SIZE, unitNameMap, NUM_UNITS, reverseUnitIndexMap, Statistics

# Fix crash bug on some macOS versions
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ACTIONS = NUM_UNITS


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
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(100))
        # layers.append(nn.Linear(100, 100))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(100))
        # layers.append(nn.Linear(100, 100))
        # layers.append(nn.LeakyReLU())
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


class A2CNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        layers.append(nn.Linear(TENSOR_INPUT_SIZE, 64))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(64))
        layers.append(nn.Linear(64, 64))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(64))
        layers.append(nn.Linear(64, 64))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(64))
        # layers.append(nn.Linear(64, 64))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(64))
        # layers.append(nn.Linear(64, 64))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.Linear(64, 64))
        # layers.append(nn.Linear(64, 64))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(64))
        self.seq = nn.Sequential(*layers)

        self.value_lin = nn.Linear(64, 1)
        self.action_lin = nn.Linear(64, NUM_ACTIONS)
        self.action_max = nn.LogSoftmax(dim=1)

    def forward(self, inputTensor):
        x = self.seq(inputTensor)
        action_log_probs = self.action_max(self.action_lin(x))  # B x NUM_ACTIONS
        state_values = self.value_lin(x)  # B
        return action_log_probs, state_values.squeeze(1)


def predict(s, debug, randomSample):
    session = json.loads(s)
    # print(state)
    stateTensor = buildOrderLoader.createState(session["states"][0])
    goalTensor = buildOrderLoader.createGoalTensor(session["goal"])
    t = buildOrderLoader.combineStateAndGoal(stateTensor, goalTensor)

    if debug:
        np.set_printoptions(precision=2, linewidth=400)
        print(t.numpy())

    best_action, state_value = trainer.find_best_action(t, explore=randomSample, eps=eps())

    global steps_done
    if randomSample:
        steps_done += 1

    return reverseUnitIndexMap[best_action], state_value


def calculate_session_rewards(s):
    return buildOrderLoader.calculate_session_rewards(s)


def loadSessionFromJson(s):
    buildOrderLoader.loadSession(s, test_memory if random.uniform(0, 1) < 0.1 else memory, statistics)


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
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
        buildOrderLoader.loadSession(s, test_memory if random.uniform(0, 1) < 0.1 else memory, statistics)
        if optimization_steps_per_load > 0:
            optimize(optimization_steps_per_load)
    print("Done")


statistics = Statistics()
data_path = "training_data/buildorders/1"
BATCH_SIZE = 512
GAMMA_PER_SECOND = 0.993
# GAMMA_PER_SECOND = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 40000
TEMPERATURE_START = 10
TEMPERATURE_END = 1
TEMPERATURE_DECAY = 8000
TARGET_UPDATE = 20

method = 'a2c'  # a2c | qlearn

if method == 'qlearn':
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00004)
    trainer = QLearning(policy_net, target_net, optimizer, device, GAMMA_PER_SECOND,TENSOR_INPUT_SIZE)
    memory = ReplayMemory(500000, prioritized_replay=True)
    test_memory = ReplayMemory(10000, prioritized_replay=False)
elif method == 'a2c':
    policy_net = A2CNetwork().to(device)
    target_net = A2CNetwork().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.00004)
    trainer = A2C(policy_net, target_net, optimizer, device, GAMMA_PER_SECOND,TENSOR_INPUT_SIZE)
    memory = ReplayMemory(2000, prioritized_replay=True)
    test_memory = ReplayMemory(10000, prioritized_replay=False)
else:
    raise Exception()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

qvalue_range = []
buildOrderLoader = BuildOrderLoader(GAMMA_PER_SECOND)

steps_done = 0


def eps():
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)


def temperature():
    return TEMPERATURE_END + (TEMPERATURE_START - TEMPERATURE_END) * math.exp(-steps_done / TEMPERATURE_DECAY)


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
        soft = np.exp((soft - soft.max()) / temp)
        soft = soft / np.sum(soft)

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


def optimize_model():
    global episode_index
    episode_index += 1

    if len(memory) < BATCH_SIZE:
        return

    loss, expected_values = trainer.train(memory, BATCH_SIZE)

    qvalue_range.append((episode_index, expected_values.numpy().mean(), expected_values.numpy().std()))
    # print(loss, np.mean(expected_values.numpy()))
    # print(loss)
    losses.append([episode_index, loss])


def evaluate_test_loss():
    loss = trainer.test(test_memory)
    test_losses.append([episode_index, loss])


plt.ioff()
episode = 0

epss = []
temps = []


def plot_loss():
    if len(test_losses) == 0:
        return

    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(2, 3, 1)
    plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    # ax.set_yscale('log')
    losses_n = np.array(losses)
    plt.plot(losses_n[:, 0], losses_n[:, 1])
    losses_n = np.array(test_losses)
    plt.plot(losses_n[:, 0], losses_n[:, 1])

    ax = fig.add_subplot(2, 3, 2)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Q value')
    qvalue_range_n = np.array(qvalue_range)
    plt.plot(qvalue_range_n[:, 0], qvalue_range_n[:, 1] - qvalue_range_n[:, 2])
    plt.plot(qvalue_range_n[:, 0], qvalue_range_n[:, 1] + qvalue_range_n[:, 2])

    x = torch.tensor(statistics.total_rewards, dtype=torch.float)
    smooth = np.convolve(x.numpy(), np.ones((10,)) / 10, mode='valid')
    ax = fig.add_subplot(2, 3, 3)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(x.numpy())
    plt.plot(smooth)

    ax = fig.add_subplot(2, 3, 4)
    # Use the last 50% of the actions
    action_slice = statistics.all_actions[len(statistics.all_actions) // 2:]
    counts = np.bincount(action_slice)
    plt.bar(range(len(counts)), counts)
    plt.xlabel('Action')
    plt.ylabel('Times Chosen')

    if memory.prioritized_replay:
        ax = fig.add_subplot(2, 3, 5)
        counts = [len(bucket) for bucket in memory.error_buckets]
        plt.bar(range(len(counts)), counts)
        plt.xlabel('Memory Bucket')
        ax.set_yscale('log')
        plt.ylabel('Size')

    ax = fig.add_subplot(2, 3, 6)
    epss.append(eps())
    temps.append(temperature())

    plt.plot(epss, label="Epsilon")
    # plt.plot(temps, label="Temperature")
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


def save(epoch):
    torch.save(policy_net.state_dict(), "models/buildorders_qlearn_" + str(epoch) + ".weights")


def load_weights(file):
    policy_net.load_state_dict(torch.load(file))


if __name__ == "__main__":
    load_all(0)
    for epoch in range(10):
        print(f"Epoch {epoch}")
        optimize_epoch()

    save(epoch)
else:
    print("Loading weights")
