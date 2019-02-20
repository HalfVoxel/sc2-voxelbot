import os
import argparse
import shutil
import random
import math
from common import count_parameters, load_all, PadSequence, training_loop
import numpy as np
import torch
import torch.nn as nn
from trainer_movement_target import MovementStepper
from trainer_rnn import TrainerRNN
import matplotlib
from datetime import datetime
import inspect
from collections import namedtuple
from build_order_loader import BuildOrderLoader, Statistics
from replay_memory import ReplayMemory
from mappings import UnitLookup, terranUnits, zergUnits, protossUnits
import game_state_loader
from game_state_loader import MovementTargetTrace
from attention import AttentionModule, AttentionDecoder, SelfAttention
from tensorboardX import SummaryWriter
from dataset_folder import create_datasets
import gzip
import pickle
# from pytorch_memory_utils.gpu_mem_track import MemTracker
# Fix crash bug on some macOS versions
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Turn on non-blocking plotting mode
plt.ion()


# @torch.jit.trace
class Net(nn.Module):
    def __init__(self, num_unit_types):
        super().__init__()

        self.global_state_size = 97
        self.minimap_size = 10
        self.minimap_layers = 12
        self.gru_hidden_size = 64

        self.lin0 = nn.Linear(self.global_state_size, 64)
        self.act0 = nn.LeakyReLU()

        self.lin1 = nn.Linear(64, 64)
        self.act1 = nn.LeakyReLU()

        self.lin2 = nn.Linear(64, 64)
        self.act2 = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.3)

        self.lin3 = nn.Linear(64, 64)
        self.act3 = nn.LeakyReLU()

        self.gru = nn.GRUCell(input_size=self.lin3.out_features, hidden_size=self.gru_hidden_size)
        # self.lin_gru = nn.Linear(self.lin3.out_features, self.gru_hidden_size)

        self.conv1 = nn.Conv2d(in_channels=self.minimap_layers, out_channels=12, kernel_size=1)
        self.act_c1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1)
        self.act_c2 = nn.LeakyReLU()

        self.conv2_2 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, padding=1)
        self.act_c2_2 = nn.LeakyReLU()

        self.conv2_3 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        self.act_c2_3 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        self.act_c3 = nn.LeakyReLU()

        self.lin_c2g = nn.Linear(self.minimap_size * self.minimap_size * self.conv3.out_channels, self.lin3.out_features)

        self.embedding = torch.nn.Parameter(torch.rand((num_unit_types, 8), dtype=torch.float, requires_grad=True))

        self.lin_t0 = nn.Linear(self.embedding.size()[1], 16)
        self.act_t0 = nn.LeakyReLU()

        self.lin_g2t = nn.Linear(self.gru_hidden_size, 16)

        self.lin_t1 = nn.Linear(16, 16)
        self.act_t1 = nn.LeakyReLU()

        self.lin_t2 = nn.Linear(16, 16)
        self.act_t2 = nn.LeakyReLU()

        self.lin4 = nn.Linear(64, 64)
        self.act4 = nn.LeakyReLU()

        self.lin_t2g = nn.Linear(16, 64)

        self.lin_g2c = nn.Linear(64, 12)
        self.act_g2c = nn.LeakyReLU()

        self.conv_c4_0 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.act_c4_0 = nn.LeakyReLU()

        self.conv_c4_1 = nn.Conv2d(in_channels=12, out_channels=4, kernel_size=3, padding=1)
        self.act_c4_1 = nn.LeakyReLU()

        self.conv_c4_2 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def init_hidden_states(self, batch_size, device):
        return torch.zeros((batch_size, self.gru_hidden_size), device=device)

    def forward(self, globalState, minimap, unit_type_counts, hiddenState):
        batch_size = globalState.size()[0]

        assert globalState.size() == (batch_size, self.global_state_size), (globalState.size(), (batch_size, self.global_state_size))
        assert minimap.size() == (batch_size, self.minimap_layers, self.minimap_size, self.minimap_size), (minimap.size(), (batch_size, self.minimap_layers, self.minimap_size, self.minimap_size))

        x = globalState
        x = self.act0(self.lin0(x))
        x = self.act1(self.lin1(x))

        m = minimap
        m = self.act_c1(self.conv1(m))
        m = self.act_c2(self.conv2(m))
        m = nn.functional.dropout(m, p=0.3)
        m = self.act_c2_2(self.conv2_2(m))
        m = self.act_c2_3(self.conv2_3(m))
        m = self.act_c3(self.conv3(m))
        self.debug_last_minimap_output = m
        m3 = m.view(batch_size, -1)

        x = self.act2(self.lin2(x) + self.lin_c2g(m3))
        x = nn.functional.dropout(x, p=0.3)
        x = self.act3(self.lin3(x))
        # x = self.act3(self.lin_gru(x))
        x = self.gru(x, hiddenState)
        hiddenState = x

        # TODO: Add position info as minimap layers

        x = self.embedding.expand(batch_size, -1, -1)
        scale = unit_type_counts.view(batch_size, unit_type_counts.size()[1], 1)
        x = x * scale
        x = self.act_t0(self.lin_t0(x))
        x = self.act_t1(self.lin_t1(x))
        x = self.act_t2(self.lin_t2(x) + self.lin_g2t(hiddenState).view(batch_size, 1, -1))
        # Sum over all unit types
        x = x.sum(dim=1)

        x = self.act4(self.lin4(hiddenState) + self.lin_t2g(x))
        x = self.act_c4_0(self.conv_c4_0(m) + self.lin_g2c(x).view(batch_size, -1, 1, 1))
        x = nn.functional.dropout2d(x, p=0.3)
        x = self.act_c4_1(self.conv_c4_1(x))
        x = self.conv_c4_2(x).view(batch_size, self.minimap_size * self.minimap_size)

        x = self.logsoftmax(x)

        # r = torch.zeros((batch_size, num_units, 16)).cuda()
        # r = self.act_u3(self.lin_u3(r))

        return x, hiddenState


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

module_name = "movement_target"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
small_input = False

print("Loading")
data_paths = ["training_data/replays/7", "training_data/replays/8", "training_data/replays/9"]
cache_dir = f"training_cache/{module_name}"

# unit_lookup = UnitLookup(terranUnits + zergUnits + protossUnits)
unit_lookup = UnitLookup(protossUnits)
buildOrderLoader = BuildOrderLoader(unit_lookup, 1.0)

# gpu_tracker.track()

model = Net(unit_lookup.num_units)
model = model.to(device)

learning_rate = 0.004
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_split = 0.1
statistics = Statistics()
batch_size = 64
current_step = 0
# gpu_tracker.track()

training_losses = []
test_losses = []
trainer = TrainerRNN(model, optimizer, action_loss_weights=[1, 1], device=device, sampleClass=MovementTargetTrace, stepperClass=MovementStepper, step_size=1)
learning_rate_decay = 200


def visualize(epoch):
    load_weights(f"models/{module_name}_{epoch}.weights")
    memory, test_memory = create_datasets(cache_dir, test_split)
    for sampleIndex in range(1000):
        sample = memory[sampleIndex]
        with gzip.open(sample.data_path, "rb") as f:
            session = pickle.load(f)
        stepper = trainer.stepperClass(model, MovementTargetTrace(*[[x] for x in sample]), device, 0, lambda a, b: 0, step_size=1)
        map_size = game_state_loader.find_map_size(session)
        while True:
            timestep = stepper.timestep
            res = stepper.step()
            if res is None:
                break

            outputs = stepper.outputs.detach().cpu().exp()
            playerIndex = sample.playerID - 1
            units = session["observations"][playerIndex]["rawUnits"][timestep]["units"]
            our_units = [u for u in units if u["owner"] == sample.playerID and game_state_loader.isMovableUnit(u, buildOrderLoader)]
            our_buildings = [u for u in units if u["owner"] == sample.playerID and not game_state_loader.isMovableUnit(u, buildOrderLoader)]

            mirror = sample.playerID == 2
            coords = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in our_units])
            coords2 = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in our_buildings])

            fig, axs = plt.subplots(nrows=4, ncols=7, figsize=(16, 8), num=1, clear=True)

            plt.sca(axs[0, 0])
            edgecolors = [(0, 0, 0, 1) if unit_lookup.military_units_mask[unit_lookup.unit_index_map[u["unit_type"]]] else (0, 0, 0, 0) for u in our_units]

            enemy_units = [u for u in units if u["owner"] != sample.playerID and u["unit_type"] in unit_lookup.unit_index_map]
            enemy_coords = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in enemy_units]).reshape(-1, 2)
            plt.scatter(enemy_coords[:, 0], enemy_coords[:, 1], c="#e41a1c")
            plt.scatter(coords2[:, 0], coords2[:, 1], c="#999999")
            plt.scatter(coords[:, 0], coords[:, 1], cmap=plt.cm.viridis, edgecolors=edgecolors)

            plt.xlim((-0.1, 1.1))
            plt.ylim((-0.1, 1.1))
            # plt.colorbar()
            plt.clim((0, 1))
            axs[0, 0].set(aspect='equal')

            plt.sca(axs[1, 0])
            plt.scatter(enemy_coords[:, 0], enemy_coords[:, 1], c="#e41a1c")
            # plt.scatter(coords[:, 0], coords[:, 1], c=sample.movement[timestep], cmap=plt.cm.viridis)

            plt.xlim((-0.1, 1.1))
            plt.ylim((-0.1, 1.1))
            # plt.colorbar()
            plt.clim((0, 1))
            axs[1, 0].set(aspect='equal')

            minimap = model.debug_last_minimap_output.detach().cpu()

            for i in range(model.minimap_layers // 2):
                for j in range(2):
                    plt.sca(axs[j, i + 1])
                    plt.imshow(sample.minimap_states[timestep][i + j * model.minimap_layers//2].transpose(0, 1), origin="lower")

            for i in range(6):
                plt.sca(axs[2, i])
                plt.imshow(minimap[0, i].transpose(0, 1), origin="lower")

            plt.sca(axs[3, 0])
            plt.imshow(outputs[0].view(model.minimap_size, model.minimap_size).transpose(0, 1), origin="lower")
            targetCoord = sample.target_positions[timestep].argmax().item()
            xc = targetCoord // model.minimap_size
            yc = targetCoord % model.minimap_size

            w = 0.5
            plt.plot([xc + w, xc - w, xc - w, xc + w, xc + w], [yc + w, yc + w, yc - w, yc - w, yc + w], c="#FF0000")

            targetCoord = sample.minimap_states[timestep][-1].argmax().item()
            xc = targetCoord // model.minimap_size
            yc = targetCoord % model.minimap_size
            w = 0.4
            plt.plot([xc + w, xc - w, xc - w, xc + w, xc + w], [yc + w, yc + w, yc - w, yc - w, yc + w], c="#000000")

            plt.sca(axs[3, 1])
            plt.imshow(sample.target_positions[timestep].transpose(0, 1), origin="lower")

            # plt.tight_layout()
            plt.pause(0.01)
            # stepper.outputs


def cache_tenors():
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    index = 0

    def load_state(s):
        def save_sample(sample):
            nonlocal index
            index += 1
            torch.save(sample, os.path.join(cache_dir, f"{index}.pt"))
        game_state_loader.loadSessionMovementTarget(s, buildOrderLoader, save_sample, None)

    load_all(data_paths, small_input, load_state)


def learning_rate_by_time(epoch):
    return learning_rate - (learning_rate - 0.0001) * min(epoch / learning_rate_decay, 1)


assert learning_rate_by_time(0) == learning_rate
assert abs(learning_rate_by_time(learning_rate_decay) - 0.0001) < 0.00001
assert abs(learning_rate_by_time(learning_rate_decay + 100) - 0.0001) < 0.00001


print(f"Parameters: {count_parameters(model)}")


def plot(tensorboard_writer):
    tensorboard_writer.add_embedding(model.embedding.data, metadata=[u[0] for u in unit_lookup.units], global_step=current_step, tag="unit embedding")


def save_tensorboard_graph(memory, tensorboard_writer):
    sample = memory[0]
    # tensorboard_writer.add_graph(model, torch.tensor([sample.state], dtype=torch.float, device=device), True)


def train(comment):
    global training_losses_by_time, current_step
    tensorboard_writer = SummaryWriter(log_dir=f"tensorboard/{module_name}/{datetime.now():%Y-%m-%d_%H:%M} {comment}")
    memory, test_memory = create_datasets(cache_dir, test_split)
    save_tensorboard_graph(test_memory, tensorboard_writer)

    padding = PadSequence(MovementTargetTrace(states=True, replay_path=False, minimap_states=True, data_path=False, playerID=False, target_positions=True, unit_type_counts=True))
    training_generator = torch.utils.data.DataLoader(memory, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, collate_fn=padding)
    testing_generator = torch.utils.data.DataLoader(test_memory, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, collate_fn=padding)

    for epoch, current_step in training_loop(training_generator, testing_generator, trainer, tensorboard_writer):
        for g in optimizer.param_groups:
            g['lr'] = learning_rate_by_time(epoch)
            print("Set learning rate to ", g['lr'])

        if epoch > 1:
            save(epoch)

            if (epoch % 5) == 0:
                plot(tensorboard_writer)


def save(epoch):
    torch.save(model.state_dict(), f"models/{module_name}_" + str(epoch) + ".weights")


def load_weights(file):
    model.load_state_dict(torch.load(file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--epoch", default=None, type=int)
    parser.add_argument("--comment", default=None, type=str)
    args = parser.parse_args()

    if args.save_cache:
        cache_tenors()

    if args.train:
        if args.comment is None or len(args.comment) == 0:
            print("You need to supply a comment for the training run (--comment)")
            exit(1)
        train(args.comment)

    if args.visualize:
        visualize(args.epoch)
