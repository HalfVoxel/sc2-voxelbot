import os
import random
import math
import time

import numpy as np
from build_order_loader import BuildOrderLoader, TENSOR_INPUT_SIZE, NUM_UNITS, Statistics, reverseUnitIndexMap, unitNameMap, unitIndexMap, nonMilitaryIndexMap
from replay_memory import ReplayMemory, Transition

import torch
import torch.nn as nn
from predict_buildorder import PredictBuildOrder
import re
import matplotlib
import sys
import json
sys.path.append("build/bin")
sys.path.append("../../../build/bin")

import cppbot
from cppbot import EnvironmentState

# Fix crash bug on some macOS versions
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


cppbot.setUnitIndexMapping(unitIndexMap, nonMilitaryIndexMap)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ACTIONS = NUM_UNITS


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        layers.append(nn.Linear(TENSOR_INPUT_SIZE, 64))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(64))
        layers.append(nn.Linear(64, 64))
        layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(64, 64))
        layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(64, 64))
        layers.append(nn.LeakyReLU())

        # layers.append(nn.BatchNorm1d(128))
        # layers.append(nn.Linear(128, 128))
        # layers.append(nn.LeakyReLU())

        # layers.append(nn.BatchNorm1d(64))
        layers.append(nn.Linear(64, ACTIONS))
        layers.append(nn.LogSoftmax(dim=1))
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

    def forward(self, inputTensor):
        x = self.seq(inputTensor)
        return x


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
    # fs = fs[:50]
    random.shuffle(fs)
    for i in range(len(fs)):
        print(f"\r{i}/{len(fs)}", end="")
        p = fs[i]
        f = open(data_path + "/" + p)
        s = f.read()
        f.close()
        buildOrderLoader.loadSession(s, test_memory if random.uniform(0, 1) < test_split else memory, statistics)
    print("Done")


print("Loading")
data_path = "training_data/buildorders/1"
model = Net()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_split = 0.1
memory = ReplayMemory(5000000, prioritized_replay=False)
test_memory = ReplayMemory(5000000, prioritized_replay=False)
buildOrderLoader = BuildOrderLoader(1.0)
statistics = Statistics()
trainer = None
batch_size = 512

training_losses = []
test_losses = []


def plot():
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    losses = np.array(training_losses)
    plt.plot(losses[:, 0], losses[:, 1])
    losses = np.array(test_losses)
    plt.plot(losses[:, 0], losses[:, 1])

    plt.pause(0.001)  # pause a bit so that plots are updated


def diagnose():
    samples = memory.get_all()
    samples2 = []
    for s in samples:
        goal = s.state[-NUM_UNITS:]
        if goal.sum() == 4 and goal[1] == 4:
            samples2.append(s)

    for sample in samples2:
        # sample = memory.get_all()
        state = sample.state
        goal = state[-NUM_UNITS:]
        if goal.sum() == 0:
            print("Skipping empty goal")
            continue

        action = sample.action
        # for i in range(NUM_UNITS):
        #     if goal[i] > 0:
        #         print(f"{unitNameMap[reverseUnitIndexMap[i]]}: {goal[i]}")

        print(f"Action: {unitNameMap[reverseUnitIndexMap[action]]}")
        # break


def diagnose2():
    epoch = 50
    load_weights("models/buildorders_4e" + str(epoch) + ".weights")
    trainer = PredictBuildOrder(model, optimizer, action_loss_weights=None)

    envs = [EnvironmentState() for i in range(10)]
    goals = []
    for env in envs:
        env.resetToGameStartConfig()
        goal = buildOrderLoader.createGoalTensor([{"type": 33, "count": 2}, {"type": 48, "count": 10}, {"type": 51, "count": 2}, {"type": 54, "count": 5}])
        env.setGoal(goal)
        goals.append(goal)

    t0 = time.time()
    for i in range(5):
        w1 = time.time()
        # jsonDatas = [json.loads(env.getState()) for env in envs]
        w2 = time.time()
        # states = [buildOrderLoader.createState(jsonData) for jsonData in jsonDatas]
        # combined_states = [buildOrderLoader.combineStateAndGoal(t, goal) for t in states]
        combined_states = [np.array(env.getObservation()) for env in envs]


        # assert (observations[0] - combined_states[0]).sum() < 0.0001

        w3 = time.time()

        dones = [s[-NUM_UNITS:].sum() == 0 for s in combined_states]
        if all(dones):
            break

        # print(goalString(state))

        w4 = time.time()
        probs = trainer.eval_batch(combined_states)
        w5 = time.time()

        # if k == 2:
        #     action = np.argmax(np.exp(probs))
        # else:
        actions = [np.random.choice(ACTIONS, p=np.exp(probs[i,:])) for i in range(len(envs))]

        for i in range(len(envs)):
            if not dones[i]:
                action = reverseUnitIndexMap[actions[i]]
                envs[i].step(action)

        w6 = time.time()
        # print((w2 - w1)*1000, (w3 - w2)*1000, (w4 - w3)*1000, (w5 - w4)*1000, (w6 - w5)*100)

    for env in envs:
        pass
        # env.print()

    t1 = time.time()
    print("TIME: ", (t1 - t0)*1000)


def diagnose3():
    epoch = 22
    load_weights("models/buildorders_4e" + str(epoch) + ".weights")
    trainer = PredictBuildOrder(model, optimizer, action_loss_weights=None)
    env = EnvironmentState()
    env.resetToGameStartConfig()
    goal = buildOrderLoader.createGoalTensor([{"type": 32, "count": 0}, {"type": 48, "count": 0}, {"type": 51, "count": 4}, {"type": 484, "count": 0}])
    env.setGoal(goal)

    while True:
        combined_state = np.array(env.getObservation())
        print(combined_state)
        probs = trainer.eval_batch([combined_state])[0, :]
        for i in range(NUM_UNITS):
            if math.exp(probs[i]) > 0.1:
                print(f"Suggested: {unitNameMap[reverseUnitIndexMap[i]]} with p={math.exp(probs[i])}")

        env.print()

        while True:
            action = input()
            found = False
            for k, v in unitNameMap.items():
                if action == v:
                    env.step(k)
                    found = True
                    break

            if found:
                break
            else:
                print(f"Unknown unit '{action}'")
                pass



def goalString(state):
    goal = state[-NUM_UNITS:]
    if goal.sum() == 0:
        return ""

    s = ""
    for i in range(NUM_UNITS):
        if goal[i] > 0:
            s += f"({unitNameMap[reverseUnitIndexMap[i]]}: {goal[i]}) "

    return s


def load_stuff():
    global trainer
    load_all(0)

    action_counts = np.zeros(ACTIONS)
    for t in memory.get_all():
        action_counts[t.action] += 1

    action_counts /= action_counts.sum()
    action_loss_weights = 1 / (ACTIONS * np.sqrt(action_counts) + 0.0001)
    print(action_loss_weights)

    # trainer = PredictBuildOrder(model, optimizer, action_loss_weights=action_loss_weights)
    trainer = PredictBuildOrder(model, optimizer, action_loss_weights=None)

def train():
    load_stuff()
    step = 0
    batch = 0
    while True:
        batch += 1
        steps = int(len(memory) / batch_size)
        for i in range(steps):
            loss = trainer.train(memory, batch_size)
            print(f"\rTraining Batch {batch} [{i}/{steps}] loss={loss}", end="")
            training_losses.append((step, loss))
            step += 1

        print()

        if len(test_memory) > 0:
            loss = trainer.test(test_memory)
            print("Test loss:", loss)
            test_losses.append((step, loss))
            plot()

            for k in range(3):
                env = EnvironmentState()
                env.resetToGameStartConfig()
                goal = buildOrderLoader.createGoalTensor([{"type": 48, "count": 10}, {"type": 55, "count": 4}])
                for i in range(50):
                    t = buildOrderLoader.createState(json.loads(env.getState()))
                    state = buildOrderLoader.combineStateAndGoal(t, goal)

                    if goalString(state) == "":
                        break

                    print(goalString(state))

                    probs = trainer.eval(Transition(state=state, action=0, next_state=None, next_action=None, reward=None, deltaTime=None))

                    if k == 2:
                        action = np.argmax(np.exp(probs))
                    else:
                        action = np.random.choice(ACTIONS, p=np.exp(probs))

                    action = reverseUnitIndexMap[action]
                    env.step(action)

                env.print()

        save(batch)


def save(epoch):
    torch.save(model.state_dict(), "models/buildorders_4e" + str(epoch) + ".weights")


def load_weights(file):
    model.load_state_dict(torch.load(file))


if __name__ == "__main__":
    diagnose2()
    # train()
