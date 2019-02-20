import os
import random
import imageio
import math
from common import count_parameters, split_into_batches, load_all, load_cached_tensors, save_cache
import time
import numpy as np
import torch
import torch.nn as nn
from trainer_buildorder import PredictBuildOrder
import re
import matplotlib
from datetime import datetime
import json
import gzip
import pickle
import inspect
from collections import defaultdict, namedtuple
import visdom
# from pytorch_memory_utils.gpu_mem_track import MemTracker
# Fix crash bug on some macOS versions
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from build_order_loader import BuildOrderLoader, Statistics
from replay_memory import ReplayMemory, Transition
from mappings import UnitLookup, terranUnits, zergUnits, protossUnits
import game_state_loader
from attention import AttentionModule, AttentionDecoder, SelfAttention
from tensorboardX import SummaryWriter

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Turn on non-blocking plotting mode
plt.ion()
tensorboard_writer = SummaryWriter(log_dir=f"tensorboard/combatpredictor/{datetime.now():%Y-%m-%d_%H:%M}")

# @torch.jit.trace


class Net(nn.Module):
    def __init__(self, num_units):
        super().__init__()

        self.num_units = num_units
        self.embedding = torch.nn.Parameter(torch.rand((num_units, 4), dtype=torch.float, requires_grad=True))

        self.lin0 = nn.Linear(self.embedding.size()[1], 32)
        self.act0 = nn.LeakyReLU()

        self.lin1 = nn.Linear(32, 32)
        self.act1 = nn.LeakyReLU()

        self.lin2 = nn.Linear(32, 32)
        self.act2 = nn.LeakyReLU()

        self.lin21 = nn.Linear(32, 32)
        self.act21 = nn.LeakyReLU()

        self.lin22 = nn.Linear(32, 32)
        self.act22 = nn.LeakyReLU()

        self.lin23 = nn.Linear(32, 32)
        self.act23 = nn.LeakyReLU()

        self.lin24 = nn.Linear(32, 32)
        self.act24 = nn.LeakyReLU()

        self.lin3 = nn.Linear(self.embedding.size()[1]*2, 32)
        self.act3 = nn.LeakyReLU()

        self.lin4 = nn.Linear(32, 2)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputTensor):
        batch_size = inputTensor.size()[0]

        # x = torch.cat([self.embedding.expand(batch_size, 2, -1, -1), inputTensor], dim=3)
        x = self.embedding.expand(batch_size, 2, -1, -1)
        scale = inputTensor[:, :, :, 6].view(batch_size, 2, -1, 1)
        x = x * scale
        # x = self.act0(self.lin0(x))
        # x = self.act1(x + self.lin1(x))
        # x = self.act2(x + self.lin2(x))

        # Sum over all units
        x = x.sum(dim=2)

        # Flatten both players
        x = x.view((batch_size, -1))
        x = self.act3(self.lin3(x))
        x = self.act21(x + self.lin21(x))
        x = self.act22(x + self.lin22(x))
        x = self.act23(x + self.lin23(x))
        x = self.act24(x + self.lin24(x))
        x = self.lin4(x)
        x = self.softmax(x)
        return x


frame = inspect.currentframe()          # define a frame to track
# gpu_tracker = MemTracker(frame)         # define a GPU tracker

unitSet = {
    "TERRAN_BANSHEE",
    "TERRAN_BATTLECRUISER",
    "TERRAN_CYCLONE",
    "TERRAN_GHOST",
    "TERRAN_HELLION",
    "TERRAN_HELLIONTANK",
    "TERRAN_LIBERATOR",
    "TERRAN_MARAUDER",
    "TERRAN_MARINE",
    "TERRAN_MEDIVAC",
    "TERRAN_MISSILETURRET",
    "TERRAN_RAVEN",
    "TERRAN_REAPER",
    "TERRAN_SCV",
    "TERRAN_SIEGETANK",
    "TERRAN_THOR",
    "TERRAN_VIKINGFIGHTER",
    "TERRAN_WIDOWMINE",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
small_input = False
load_cached_data = True
caching_filename = "cached_tensors_combat.pickle"
if small_input:
    caching_filename = "cached_tensors_combat_small.pickle"

print("Loading")
data_paths = ["training_data/combatsimulations/1"]
# unit_lookup = UnitLookup(terranUnits + zergUnits + protossUnits)
unit_lookup = UnitLookup([u for u in terranUnits if u[0] in unitSet])

# gpu_tracker.track()

# model = Net((2, tensor_input_size), num_units=unit_lookup.num_units)
model = Net(num_units=unit_lookup.num_units)
model = model.to(device)

learning_rate = 0.004
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_split = 0.1
memory = ReplayMemory(5000000, prioritized_replay=False)
test_memory = ReplayMemory(5000000, prioritized_replay=False)
statistics = Statistics()
trainer = None
batch_size = 2048
current_step = 0
# gpu_tracker.track()

training_losses = []
test_losses = []
learning_rate_decay_epoch = 500


def learning_rate_by_time(epoch):
    return learning_rate - (learning_rate - 0.0001) * min(epoch / learning_rate_decay_epoch, 1)


assert learning_rate_by_time(0) == learning_rate
assert abs(learning_rate_by_time(learning_rate_decay_epoch) - 0.0001) < 0.00001
assert abs(learning_rate_by_time(learning_rate_decay_epoch) - 0.0001) < 0.00001


print(f"Parameters: {count_parameters(model)}")


def plot():
    weights = model.embedding.data.cpu().numpy()
    fig, axs = plt.subplots(weights.shape[1], 1, sharex=True, clear=True, figsize=(6, 10), num=1)
    # plt.clf()

    for i in range(weights.shape[1]):
        # ax = fig.add_subplot(1+i, 1, weights.shape[0])
        xs = list(range(weights.shape[0]))
        ys = weights[:, i]
        labels = [u[0].replace("TERRAN_", "").title() for u in unit_lookup.units]
        # print(xs, ys, labels)
        axs[i].bar(xs, ys, tick_label=labels)
        # plt.bar(xs, ys, tick_label=labels)

    plt.xticks(rotation=90)
    plt.tight_layout()
    # plt.pause(0.001)
    # plt.show()

    imgname = f'combat_{current_step}.png'
    plt.savefig(imgname)


    # plt.title(' Training/Test Loss')
    # plt.xlabel('Batch')
    # plt.ylabel('Loss')
    # plt.grid()
    # losses = np.array(training_losses)
    # plt.plot(losses[:, 0], losses[:, 1])
    # # plot_windows["loss2"] = vis.line(X=losses[:, 0], Y=losses[:, 1], win=plot_windows["loss2"], name="trace1")
    # losses = np.array(test_losses)
    # plt.plot(losses[:, 0], losses[:, 1])
    # tensorboard_writer.add_figure("embedding", fig, global_step=current_step, close=True)
    # img = imageio.imread(imgname)
    # print(type(img), np.asarray(img).dtype)
    # tensorboard_writer.add_image("embedding", np.asarray(img), global_step=current_step)

    # plt.tight_layout()
    # plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()
    tensorboard_writer.add_embedding(model.embedding.data, metadata=[
                                     u[0] for u in unit_lookup.units], global_step=current_step, tag="unit embedding")


CombatSample = namedtuple('CombatSample', ('state', 'action'))


def loadInstance(instance, memory):
    units = np.zeros((2, unit_lookup.num_units))
    for unit in instance["startingState"]["units"]:
        if not unit["type"] in unit_lookup.unit_index_map:
            raise Exception(f"Invalid unit {unit['type']}")

        unit_type = unit_lookup.unit_index_map[unit["type"]]
        units[unit["owner"] - 1, unit_type] += (unit["health"] + unit["shield"]) / \
            (unit["health_max"] + unit["shield_max"])

    healths = [0, 0]
    for u in instance["outcome"]["state"]["units"]:
        healths[u["owner"] - 1] += u["health"] + u["shield"]

    winner = 0 if healths[0] > healths[1] else 1

    onehot_units = np.zeros((2, unit_lookup.num_units, 7), dtype=np.float32)
    onehot_units[:, :, 0] = units[:, :] == 0
    onehot_units[:, :, 1] = units[:, :] >= 1
    onehot_units[:, :, 2] = units[:, :] >= 2
    onehot_units[:, :, 3] = units[:, :] >= 4
    onehot_units[:, :, 4] = units[:, :] >= 8
    onehot_units[:, :, 5] = units[:, :] >= 16
    onehot_units[:, :, 6] = units[:, :]

    memory.push(CombatSample(state=onehot_units, action=winner))
    # Mirror
    memory.push(CombatSample(state=onehot_units[[1, 0], :], action=1 - winner))


def loadSession(session, memory):
    for instance in session["instances"]:
        loadInstance(instance, memory)


def load_stuff():
    global trainer
    if load_cached_data:
        load_cached_tensors(caching_filename, memory, test_memory)
    else:
        def load_state(s):
            mem = test_memory if random.uniform(0, 1) < test_split else memory
            loadSession(s, mem)

        load_all(data_paths, small_input, load_state)
        save_cache(caching_filename, memory, test_memory)

    sample = test_memory.sample(1)[0]
    tensorboard_writer.add_graph(model, torch.tensor([sample.state], dtype=torch.float, device=device), True)
    # trainer = PredictBuildOrder(model, optimizer, action_loss_weights=action_loss_weights)
    trainer = PredictBuildOrder(model, optimizer, action_loss_weights=None, device=device, sampleClass=CombatSample)


def train():
    global training_losses_by_time, current_step
    load_stuff()
    step = 0
    epoch = 0

    training_generator = torch.utils.data.DataLoader(memory, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)
    testing_generator = torch.utils.data.DataLoader(test_memory, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)

    while True:
        for g in optimizer.param_groups:
            g['lr'] = learning_rate_by_time(epoch)
            print("Set learning rate to ", g['lr'])

        epoch += 1
        print(f"\rTraining Epoch {epoch}", end="")
        for i, batch_tuple in enumerate(training_generator):
            batch = CombatSample(*batch_tuple)
            loss = trainer.train2(batch)
            print(f"\rTraining Epoch {epoch} [{i+1}/{len(memory) // batch_size}] loss={loss}", end="")
            training_losses.append((step, loss))
            step += 1
            current_step = step
            tensorboard_writer.add_scalar("training loss", loss, step)

        print()
        if len(test_memory) > 0:
            trainer.time_threshold = 0
            total_loss = 0
            total_weight = 0
            for i, batch_tuple in enumerate(testing_generator):
                batch = CombatSample(*batch_tuple)
                loss = trainer.test2(batch)
                total_weight += batch.state.size()[0]
                total_loss += loss * batch.state.size()[0]

            total_loss /= total_weight

            print("Test loss:", total_loss)
            test_losses.append((step, total_loss))
            tensorboard_writer.add_scalar("test loss", total_loss, step)

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
        "replayInfo": {"replay_path": ""},
        "mmrs": [10000, 10000]
    }
    json.loads(json_state)
