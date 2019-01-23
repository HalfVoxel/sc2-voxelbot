import os
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
from trainer_win import TrainerWinPredictor
import re
import matplotlib
import json
import gzip
import pickle
# Fix crash bug on some macOS versions
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from build_order_loader import BuildOrderLoader, Statistics
from replay_memory import ReplayMemory, Transition
from mappings import UnitLookup, terranUnits, zergUnits, protossUnits
import game_state_loader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Turn on non-blocking plotting mode
plt.ion()


class Net(nn.Module):
    def __init__(self, input_shape, num_units):
        super().__init__()

        layers = []
        print("Input shape", input_shape)
        layers.append(nn.Linear(input_shape[-1], 32))
        layers.append(nn.Tanh())
        # layers.append(nn.BatchNorm1d(64))
        # layers.append(nn.Linear(128, 32))
        # layers.append(nn.Tanh())

        self.gru_hidden_size = 32
        self.gru = nn.GRUCell(input_size=64 + 32, hidden_size=self.gru_hidden_size)
        self.lin1 = nn.Linear(32, 32)
        self.act = nn.Tanh()

        self.lin2 = nn.Linear(32, 32)

        # layers.append(nn.BatchNorm1d(128))
        # layers.append(nn.Linear(128, 128))
        # layers.append(nn.Tanh())

        # layers.append(nn.BatchNorm1d(64))
        self.lin3 = nn.Linear(32, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        # layers.append(nn.BatchNorm1d(64))
        # layers.append(nn.Linear(64, 64))
        # layers.append(nn.Tanh())
        # layers.append(nn.BatchNorm1d(64))
        # layers.append(nn.Linear(64, 64))
        # layers.append(nn.Tanh())
        # layers.append(nn.Tanh())
        # layers.append(nn.Linear(64, 64))
        # layers.append(nn.Linear(64, 64))
        # layers.append(nn.Tanh())
        # layers.append(nn.BatchNorm1d(64))
        self.seq = nn.Sequential(*layers)

        self.m_lin1 = nn.Linear(3, 4)
        self.m_lin2 = nn.Linear(2 * 25 * 4, 32)

    def forward(self, inputTensor, minimapInputTensor, stateTensors=None):
        if stateTensors is None:
            stateTensors = torch.zeros((inputTensor.size()[0], self.gru_hidden_size))

        # flattened = minimapInputTensor.view((inputTensor.size()[0], 2, 25, -1))
        x2 = self.m_lin1(minimapInputTensor)
        x2 = self.act(x2)
        x2 = x2.view((inputTensor.size()[0], -1))
        x2 = self.m_lin2(x2)
        x2 = self.act(x2)

        x = self.seq(inputTensor)
        x = x.view([inputTensor.size()[0], -1])
        x = torch.cat([x, x2], dim=1)
        stateTensors = self.gru(x, stateTensors)
        x = stateTensors
        x = self.act(self.lin1(x))
        x = self.act(self.lin2(x))
        x = self.lin3(x)
        x = self.softmax(x)
        return x, stateTensors


def natural_sort(l):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def load_cached():
    with gzip.open(caching_filename, 'rb') as f:
        print("Loading cached tensors")
        chunks = pickle.load(f)
        for i in range(chunks):
            items1 = pickle.load(f)
            for item in items1:
                memory.push(item)

        print("Loading cached test tensors")
        items2 = pickle.load(f)
        for item in items2:
            test_memory.push(item)


def load_all(optimization_steps_per_load: int):
    print("Loading training data...")
    for data_path in data_paths:
        fs = os.listdir(data_path)
        fs = natural_sort(fs)
        if small_input:
            fs = fs[:20]
        random.shuffle(fs)
        for i in range(len(fs)):
            print(f"\r{i}/{len(fs)}", end="")
            p = fs[i]
            try:
                with gzip.open(data_path + "/" + p, "rb") as f:
                    s = pickle.load(f)
            except Exception:
                print("Failed to load and deserialize", p)
                continue
            game_state_loader.loadSession(s, buildOrderLoader, test_memory if random.uniform(0, 1) < test_split else memory, statistics)
        print("Done")

    with gzip.open(caching_filename, 'wb') as f:
        batches = list(split_into_batches(memory.get_all(), 64))
        pickle.dump(len(batches), f, protocol=pickle.HIGHEST_PROTOCOL)
        for chunk in batches:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_memory.get_all(), f, protocol=pickle.HIGHEST_PROTOCOL)


small_input = False
caching_filename = "cached_tensors.pickle"
if small_input:
    caching_filename = "cached_tensors_small.pickle"

print("Loading")
data_paths = ["training_data/replays/2", "training_data/replays/3", "training_data/replays/4", "training_data/replays/5", "training_data/replays/6"]
unit_lookup = UnitLookup(terranUnits + zergUnits + protossUnits)
buildOrderLoader = BuildOrderLoader(unit_lookup, 1.0)

tensor_input_size = game_state_loader.getInputTensorSize(buildOrderLoader)
model = Net((2, tensor_input_size), num_units=unit_lookup.num_units)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_split = 0.1
memory = ReplayMemory(5000000, prioritized_replay=False)
test_memory = ReplayMemory(5000000, prioritized_replay=False)
statistics = Statistics()
trainer = None
batch_size = 128

training_losses = []
training_losses_by_time = []
test_losses = []
test_losses_by_time = []


def split_into_batches(l, batch_size):
    """
    :param l:           list
    :param batch_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), batch_size):
        yield l[i:min(len(l), i+batch_size)]


def plot():
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(2, 2, 1)
    plt.title(' Training/Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    losses = np.array(training_losses)
    ax.plot(losses[:, 0], losses[:, 1])
    losses = np.array(test_losses)
    ax.plot(losses[:, 0], losses[:, 1])

    for i in range(2):
        ax = fig.add_subplot(2, 2, 3 + i)
        source = test_losses_by_time[-1] if i == 0 else training_losses_by_time
        losses = np.array([(x[0], x[1]) for x in source])
        xs = losses[:,0]
        probs = np.exp(-losses[:,1])

        bins = 10
        means = np.zeros(bins)
        means_xs = np.zeros(bins)
        for bin_index in range(bins):
            means_xs[bin_index] = (bin_index+0.5)/bins
            means[bin_index] = np.mean(probs[np.where((xs >= bin_index/bins) & (xs <= (bin_index+1)/bins))])

        worst_index = np.argmin(probs * ((1 - xs) + 0.2))
        worst_trace = source[worst_index][2]
        mask_values = [1 if (x[2].replay_path == worst_trace.replay_path) else 0 for x in source]
        mask = np.array(mask_values)
        mask_indices = np.where(mask)
        print(f"Worst replay: {worst_trace.replay_path}")

        plt.title("Correct outcome (test)" if i == 0 else "Correct outcome (train)")
        plt.xlabel('Time')
        plt.ylabel('Probability')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])

        ax.scatter(xs, probs, marker='.', alpha=0.3 if i==0 else 0.2, s=1, color="#377eb8")
        ax.scatter(losses[mask_indices,0], probs[mask_indices], marker='.', alpha=0.5, s=1, color="#e41a1c")
        ax.plot(means_xs, means, color="#000000")

    plt.tight_layout()
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def load_stuff():
    global trainer
    load_all(0)
    # load_cached()

    # trainer = PredictBuildOrder(model, optimizer, action_loss_weights=action_loss_weights)
    trainer = TrainerWinPredictor(model, optimizer, action_loss_weights=None)


def train():
    global training_losses_by_time
    load_stuff()
    step = 0
    epoch = 0
    while True:
        epoch += 1
        trainer.time_threshold = math.exp(-epoch/5.0)
        training_losses_by_time = []
        all_samples = memory.get_all()
        random.shuffle(all_samples)
        batches = list(split_into_batches(all_samples, batch_size))
        print(f"\rTraining Epoch {epoch}", end="")
        for i in range(len(batches)):
            loss, losses_by_time = trainer.train(batches[i])
            print(f"\rTraining Epoch {epoch} [{i+1}/{len(batches)}] loss={loss}", end="")
            training_losses.append((step, loss))
            step += 1
            training_losses_by_time += losses_by_time

        print()

        if len(test_memory) > 0:
            trainer.time_threshold = 0
            loss, losses_by_time = trainer.test(test_memory)
            print("Test loss:", loss)
            test_losses.append((step, loss))
            test_losses_by_time.append(losses_by_time)

            plot()

        save(epoch)


def save(epoch):
    torch.save(model.state_dict(), "models/win_1e" + str(epoch) + ".weights")


def load_weights(file):
    model.load_state_dict(torch.load(file))


if __name__ == "__main__":
    train()
# else:
#     global trainer
#     trainer = TrainerWinPredictor(model, optimizer, action_loss_weights=None)
#     load_weights("models/win_1e22.weights")


def calculate_win_probability(json_state):
    session = {
        "replayInfo": { "replay_path": "" },
        "mmrs": [10000, 10000]
    }
    json.loads(json_state)
