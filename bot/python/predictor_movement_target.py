import os
import argparse
import shutil
import subprocess
import math
from common import count_parameters, load_all, PadSequence, training_loop, print_parameters
import common
import numpy as np
import torch
import torch.nn as nn
from trainer_movement_target import MovementStepper
from trainer_rnn import TrainerRNN
import matplotlib
from datetime import datetime
from build_order_loader import BuildOrderLoader
from mappings import UnitLookup, terranUnits, zergUnits, protossUnits
import game_state_loader
from game_state_loader import MovementTargetTrace
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
        self.minimap_size = 14
        self.minimap_layers = 12 + 2 + 4

        scale = 2
        self.gru_hidden_size = 64 * scale

        self.constant_minimap = torch.nn.Parameter(torch.randn((2, self.minimap_size, self.minimap_size), dtype=torch.float, requires_grad=True))

        self.lin0 = nn.Linear(self.global_state_size, 64 * scale)
        self.act0 = nn.LeakyReLU()

        self.lin1 = nn.Linear(64 * scale, 64 * scale)
        self.act1 = nn.LeakyReLU()

        self.lin2 = nn.Linear(64 * scale, 64)
        self.act2 = nn.LeakyReLU()

        self.dropout = nn.Dropout(p=0.3)

        self.lin3 = nn.Linear(64, 64)
        self.act3 = nn.LeakyReLU()

        self.gru = nn.GRUCell(input_size=self.lin3.out_features, hidden_size=self.gru_hidden_size)
        # self.lin_gru = nn.Linear(self.lin3.out_features, self.gru_hidden_size)

        self.conv1 = nn.Conv2d(in_channels=self.minimap_layers, out_channels=12 * scale, kernel_size=1)
        self.act_c1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=12 * scale, out_channels=12, kernel_size=3, padding=1)
        self.act_c2 = nn.LeakyReLU()

        self.conv2_2 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, padding=1)
        self.act_c2_2 = nn.LeakyReLU()

        # self.conv2_3 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        # self.act_c2_3 = nn.LeakyReLU()

        # self.conv3 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        # self.act_c3 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, padding=0)
        self.act_c3 = nn.LeakyReLU()

        self.conv_c2g_1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        self.conv_c2g_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_c2g_3 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.conv_c2g_4 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.conv_c2g_5 = nn.Conv2d(in_channels=12, out_channels=self.lin2.out_features, kernel_size=3, padding=1)
        self.conv_c2g_6 = nn.MaxPool2d(kernel_size=3, padding=0)
        self.act_c2g_1 = nn.ReLU()
        self.act_c2g_3 = nn.ReLU()
        self.act_c2g_5 = nn.ReLU()

        # self.lin_c2g = nn.Linear(self.minimap_size * self.minimap_size * self.conv3.out_channels, self.lin2.out_features)

        self.embedding = torch.nn.Parameter(torch.rand((num_unit_types, 8), dtype=torch.float, requires_grad=True))

        self.lin_t0 = nn.Linear(self.embedding.size()[1], 16)
        self.act_t0 = nn.LeakyReLU()

        self.lin_g2t = nn.Linear(self.gru_hidden_size, 16)

        self.lin_t1 = nn.Linear(16, 16)
        self.act_t1 = nn.LeakyReLU()

        self.lin_t2 = nn.Linear(16, 16)
        self.act_t2 = nn.LeakyReLU()

        self.lin4 = nn.Linear(self.gru_hidden_size, 64)
        self.act4 = nn.LeakyReLU()

        self.lin_t2g = nn.Linear(16, 64)

        self.lin_g2c = nn.Linear(64, 12)
        self.act_g2c = nn.LeakyReLU()

        self.conv_c4_0 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.act_c4_0 = nn.LeakyReLU()

        self.conv_c4_1 = nn.Conv2d(in_channels=12, out_channels=4 * scale, kernel_size=3, padding=1)
        self.act_c4_1 = nn.LeakyReLU()

        self.conv_c4_2 = nn.Conv2d(in_channels=4 * scale, out_channels=1, kernel_size=3, padding=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.xs = (torch.tensor(list(range(self.minimap_size)), dtype=torch.float) / self.minimap_size).unsqueeze(1).expand(self.minimap_size, self.minimap_size).cuda()
        self.ys = self.xs.transpose(0, 1)

        # self.conv_m1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=0, stride=3)
        # self.act_m1 = nn.LeakyReLU()


        self.conv_m1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.act_m1_1 = nn.ReLU()
        self.conv_m1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.act_m1_2 = nn.ReLU()
        self.conv_m1_3 = nn.MaxPool2d(kernel_size=3)

        self.conv_m2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.act_m2_1 = nn.ReLU()
        self.conv_m2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.act_m2_2 = nn.ReLU()
        self.conv_m2_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_m3_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.act_m3_1 = nn.ReLU()
        self.conv_m3_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.act_m3_2 = nn.ReLU()
        self.conv_m3_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_m4_1 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1, padding=0)
        self.act_m4_1 = nn.ReLU()

        # self.bn_1 = nn.BatchNorm2d(16)
        # self.bn_2 = nn.BatchNorm2d(16)
        # self.bn_3 = nn.BatchNorm2d(16)
        # self.bn_4 = nn.BatchNorm2d(16)
        # self.bn_5 = nn.BatchNorm2d(16)

        # self.bn_c1 = nn.BatchNorm2d(12)
        # self.bn_c2 = nn.BatchNorm2d(12)

        # self.bn_g = nn.BatchNorm(64)

    def init_hidden_states(self, batch_size, device):
        return torch.zeros((batch_size, self.gru_hidden_size), device=device)

    def forward_minimap(self, minimap):
        batch_size = minimap.size()[0]

        assert minimap.size() == (batch_size, 168, 168)
        minimap = minimap.view(batch_size, 1, 168, 168)
        x = minimap

        y = self.act_m1_1(self.conv_m1_1(x))
        # y = self.bn_1(y)
        x = self.act_m1_2(x + self.conv_m1_2(y))
        # x = self.bn_2(x)
        x = self.conv_m1_3(x)

        y = self.act_m2_1(self.conv_m2_1(x))
        # y = self.bn_3(y)
        x = self.act_m2_2(x + self.conv_m2_2(y))
        # x = self.bn_4(x)
        x = self.conv_m2_3(x)

        y = self.act_m3_1(self.conv_m3_1(x))
        # y = self.bn_4(y)
        x = self.act_m3_2(x + self.conv_m3_2(y))
        # x = self.bn_5(x)
        x = self.conv_m3_3(x)

        x = self.act_m4_1(self.conv_m4_1(x))
        return x

    def forward(self, globalState, minimap, unit_type_counts, hiddenState, pathfinding_minimap):
        batch_size = globalState.size()[0]

        minimap = torch.cat([self.xs.expand(batch_size, 1, -1, -1), self.ys.expand(batch_size, 1, -1, -1), pathfinding_minimap, minimap], dim=1)

        assert globalState.size() == (batch_size, self.global_state_size), (globalState.size(), (batch_size, self.global_state_size))
        assert minimap.size() == (batch_size, self.minimap_layers, self.minimap_size, self.minimap_size), (minimap.size(), (batch_size, self.minimap_layers, self.minimap_size, self.minimap_size))

        x = globalState
        x = self.act0(self.lin0(x))
        x = self.act1(self.lin1(x))

        m = minimap
        m = self.act_c1(self.conv1(m))
        m = self.act_c2(self.conv2(m))
        # m = self.bn_c1(m)
        m = nn.functional.dropout(m, p=0.1)
        m = self.act_c2_2(self.conv2_2(m))
        # m = self.act_c2_3(self.conv2_3(m))
        m = self.act_c3(self.conv3(m))
        self.debug_last_minimap_output = m
        # m3 = m.view(batch_size, -1)

        c2g = self.act_c2g_1(self.conv_c2g_1(m))
        c2g = self.conv_c2g_2(c2g)
        c2g = self.act_c2g_3(self.conv_c2g_3(c2g))
        c2g = self.conv_c2g_4(c2g)
        c2g = self.act_c2g_5(self.conv_c2g_5(c2g))
        c2g = self.conv_c2g_6(c2g).view(batch_size, 64)

        # x = self.act2(self.lin2(x) + self.lin_c2g(m3))
        x = self.act2(self.lin2(x) + c2g)
        x = nn.functional.dropout(x, p=0.1)
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
        # x = self.bn_g(x)
        x = self.act_c4_0(self.conv_c4_0(m) + self.lin_g2c(x).view(batch_size, -1, 1, 1))
        self.last_c4_0 = x
        x = nn.functional.dropout2d(x, p=0.1)
        x = self.act_c4_1(self.conv_c4_1(x))
        self.last_c4_1 = x
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

import flamegraph
# flamegraph.start_profile_thread(fd=open("./perf.log", "w"))
module_name = "movement_target"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
small_input = False

print("Loading")
data_paths = ["training_data/replays/s2"]
cache_dir = f"training_cache/{module_name}"

# unit_lookup = UnitLookup(terranUnits + zergUnits + protossUnits)
unit_lookup = UnitLookup(protossUnits)
buildOrderLoader = BuildOrderLoader(unit_lookup, 1.0)
print("A")
# gpu_tracker.track()

model = Net(unit_lookup.num_units)
print("B")
model = model.to(device)
print("C")

learning_rate = 0.004
learning_rate_decay = 500
learning_rate_final = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_split = 0.1
batch_size = 48
current_step = 0
# gpu_tracker.track()

training_losses = []
test_losses = []
trainer = TrainerRNN(model, optimizer, action_loss_weights=[1, 1], device=device, sampleClass=MovementTargetTrace, stepperClass=MovementStepper, step_size=1)
print("D")

padding = PadSequence(MovementTargetTrace(
    states='stack-timewise',
    replay_path=None,
    minimap_states='stack-timewise',
    data_path=None,
    playerID=None,
    target_positions='stack-timewise',
    unit_type_counts='stack-timewise',
    pathfinding_minimap=None
))


def visualize(epoch):
    print("E")
    load_weights(f"models/{module_name}_{epoch}.weights")
    print("F")
    memory, test_memory = create_datasets(cache_dir, test_split)
    print("G")
    for sampleIndex in range(1000):
        sample = memory[sampleIndex]
        print("H")
        with gzip.open(sample.data_path, "rb") as f:
            session = pickle.load(f)

        print("I")
        stepper = trainer.stepperClass(model, padding([sample])[0], device, 0, lambda a, b: 0, step_size=1)
        print("J")
        map_size = game_state_loader.find_map_size(session)
        print("K")
        while True:
            timestep = stepper.timestep
            print("Eval")
            model.eval()
            with torch.no_grad():
                res = stepper.step()
            if res is None:
                break

            outputs = stepper.outputs.detach().cpu().exp()

            if sample.minimap_states[timestep][-1].sum() == 0:
                print("Skipping frame")
                continue

            print("Plotting")
            playerIndex = sample.playerID - 1
            units = session["observations"][playerIndex]["rawUnits"][timestep]["units"]
            our_units = [u for u in units if u["owner"] == sample.playerID and game_state_loader.isMovableUnit(u, buildOrderLoader)]
            our_buildings = [u for u in units if u["owner"] == sample.playerID and not game_state_loader.isMovableUnit(u, buildOrderLoader)]

            mirror = False #sample.playerID == 2
            coords = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in our_units])
            coords2 = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in our_buildings])

            fig, axs = plt.subplots(nrows=6, ncols=8, figsize=(16, 8), num=1, clear=True)

            plt.sca(axs[0, 0])
            edgecolors = [(0, 0, 0, 1) if unit_lookup.military_units_mask[unit_lookup.unit_index_map[u["unit_type"]]] else (0, 0, 0, 0) for u in our_units]

            enemy_units = [u for u in units if u["owner"] != sample.playerID and u["unit_type"] in unit_lookup.unit_index_map]
            enemy_coords = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in enemy_units]).reshape(-1, 2)
            print("A")

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

            layers = sample.minimap_states[timestep].size()[0]
            for i in range(layers // 2):
                for j in range(2):
                    plt.sca(axs[j, i + 1])
                    plt.imshow(sample.minimap_states[timestep][i + j * layers // 2].transpose(0, 1), origin="lower")

            minimap = model.debug_last_minimap_output.detach().cpu()
            for i in range(6):
                plt.sca(axs[2, i])
                plt.imshow(minimap[0, i].transpose(0, 1), origin="lower")

            minimap = model.last_c4_0.detach().cpu()
            for i in range(8):
                plt.sca(axs[4, i])
                plt.imshow(minimap[0, i].transpose(0, 1), origin="lower")

            minimap = model.last_c4_1.detach().cpu()
            for i in range(8):
                plt.sca(axs[5, i])
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

            print("B")

            plt.sca(axs[3, 1])
            plt.imshow(sample.target_positions[timestep].transpose(0, 1), origin="lower")

            plt.sca(axs[3, 2])
            plt.imshow(sample.pathfinding_minimap.transpose(0, 1), origin="lower")

            # for i in range(2):
            #     plt.sca(axs[3, 2 + i])
            #     plt.imshow(model.constant_minimap[i].detach().cpu().transpose(0, 1), origin="lower")

            for i in range(4):
                plt.sca(axs[3, 3 + i])
                plt.imshow(stepper.batch.pathfinding_minimap.detach()[0, i, :, :].cpu().transpose(0, 1), origin="lower")

            print("C")
            # plt.tight_layout()
            plt.pause(0.01)
            # stepper.outputs


def cache_tensors():
    def load_state(s, target_filepath):
        def save_sample(sample):
            with gzip.open(target_filepath, 'wb') as f:
                torch.save(sample, f)
        game_state_loader.loadSessionMovementTarget(s, buildOrderLoader, save_sample, None)

    version = 0
    common.cache_tenors(data_paths, cache_dir, small_input, load_state, version)


def learning_rate_by_time(epoch):
    return learning_rate - (learning_rate - learning_rate_final) * min(epoch / learning_rate_decay, 1)


assert learning_rate_by_time(0) == learning_rate
assert abs(learning_rate_by_time(learning_rate_decay) - learning_rate_final) < 0.00001
assert abs(learning_rate_by_time(learning_rate_decay + 100) - learning_rate_final) < 0.00001


def plot(tensorboard_writer):
    tensorboard_writer.add_embedding(model.embedding.data, metadata=[u[0] for u in unit_lookup.units], global_step=current_step, tag="unit embedding")


def save_tensorboard_graph(memory, tensorboard_writer):
    # sample = memory[0]
    # tensorboard_writer.add_graph(model, torch.tensor([sample.state], dtype=torch.float, device=device), True)
    pass


def train(comment):
    print_parameters(model)
    print(f"Parameters: {count_parameters(model)}")

    global training_losses_by_time, current_step
    tensorboard_writer = SummaryWriter(log_dir=f"tensorboard/{module_name}/{datetime.now():%Y-%m-%d_%H:%M} {comment}")
    memory, test_memory = create_datasets(cache_dir, test_split)
    save_tensorboard_graph(test_memory, tensorboard_writer)

    training_generator = torch.utils.data.DataLoader(memory, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3, collate_fn=padding)
    testing_generator = torch.utils.data.DataLoader(test_memory, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1, collate_fn=padding)

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
    model.load_state_dict(torch.load(file, map_location='cpu'))

class Stepper:
    def __init__(self):
        self.stepper = trainer.stepperClass(model, None, device, 0, lambda a, b: 0, step_size=1)

    def step(self, json_data, unit_tag_mask, playerID):
        observer_session = json.loads(json_data)
        trace = game_state_loader.loadSessionMovementTarget2(observationSession, playerID, loader, unit_tag_mask, "invalid"):
        self.stepper.set_batch(padding([trace])[0])
        self.stepper.init_hidden_states()
        self.stepper.step()
        result = self.stepper.outputs.detach().cpu().exp().numpy()[0, :, 1].tolist()
        print(result)
        return result

if __name__ == "__main__":
    common.train_interface(cache_tensors, train, visualize)
