print("0")
import os
import json
import common
from common import count_parameters, load_all, PadSequence, training_loop
import numpy as np
import torch
import torch.nn as nn
print("A")
from trainer_movement import MovementStepper
from trainer_rnn import TrainerRNN
from datetime import datetime
from build_order_loader import BuildOrderLoader, Statistics
from mappings import UnitLookup, terranUnits, zergUnits, protossUnits
import game_state_loader
from game_state_loader import MovementTrace
print("B")
print("C!")
from dataset_folder import create_datasets
import gzip
import pickle
print("C")
# from pytorch_memory_utils.gpu_mem_track import MemTracker

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# @torch.jit.trace
class Net(nn.Module):
    def __init__(self, num_unit_types):
        super().__init__()

        self.unit_size = 22
        self.global_state_size = 97
        self.minimap_size = 10
        self.minimap_layers = 10
        self.gru_hidden_size = 128

        self.embedding = nn.Embedding(num_unit_types, 8)

        self.lin0 = nn.Linear(self.global_state_size, 32)
        self.act0 = nn.LeakyReLU()

        self.lin1 = nn.Linear(32, 64)
        self.act1 = nn.LeakyReLU()

        self.lin2 = nn.Linear(64, 64)
        self.act2 = nn.LeakyReLU()

        self.lin3 = nn.Linear(64, 64)
        self.act3 = nn.LeakyReLU()

        self.gru = nn.GRUCell(input_size=self.lin3.out_features, hidden_size=self.gru_hidden_size)
        # self.lin_gru = nn.Linear(self.lin3.out_features, self.gru_hidden_size)

        self.lin_u1 = nn.Linear(self.unit_size - 1 + self.embedding.embedding_dim, 16)
        self.act_u1 = nn.LeakyReLU()
        self.lin_u2 = nn.Linear(16, 32)
        self.act_u2 = nn.LeakyReLU()
        self.lin_u2_2 = nn.Linear(32, 16)
        self.act_u2_2 = nn.LeakyReLU()
        self.lin_u2_3 = nn.Linear(16, 16)
        self.act_u2_3 = nn.LeakyReLU()
        self.lin_u3 = nn.Linear(16, 2)
        self.act_u3 = nn.LogSoftmax(dim=2)

        self.lin_g2u = nn.Linear(self.gru_hidden_size, self.lin_u2_2.in_features)
        self.act_u2 = nn.LeakyReLU()

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
        self.lin_c2u = nn.Linear(self.conv3.out_channels, self.lin_u2_2.in_features)

    def init_hidden_states(self, batch_size, device):
        return torch.zeros((batch_size, self.gru_hidden_size), device=device)

    def forward(self, globalState, minimap, rawUnits, hiddenState, rawUnitCoords):
        batch_size = globalState.size()[0]
        num_units = rawUnits.size()[1]

        assert rawUnits.size() == (batch_size, num_units, self.unit_size), (rawUnits.size(), (batch_size, num_units, self.unit_size))
        assert globalState.size() == (batch_size, self.global_state_size), (globalState.size(), (batch_size, self.global_state_size))
        assert minimap.size() == (batch_size, self.minimap_layers, self.minimap_size, self.minimap_size), (minimap.size(), (batch_size, self.minimap_layers, self.minimap_size, self.minimap_size))

        x = globalState
        x = self.act0(self.lin0(x))
        x = self.act1(self.lin1(x))

        m = minimap
        m = self.act_c1(self.conv1(m))
        m = self.act_c2(self.conv2(m))
        m = self.act_c2_2(self.conv2_2(m))
        m = self.act_c2_3(self.conv2_3(m))
        m = self.act_c3(self.conv3(m))
        self.debug_last_minimap_output = m
        # Note: minimap is indexed using [x,y]
        unitCoordinateIndices = rawUnitCoords[:, :, 0] * self.minimap_size + rawUnitCoords[:, :, 1]
        unitCoordinateIndices = unitCoordinateIndices.unsqueeze(-1).expand(-1, -1, m.size()[1])
        m2 = m.view(batch_size, -1, self.minimap_size * self.minimap_size).transpose(1, 2)
        minimapUnitInfo = m2.gather(index=unitCoordinateIndices, dim=1)
        m3 = m.view(batch_size, -1)

        x = self.act2(self.lin2(x) + self.lin_c2g(m3))
        x = self.act3(self.lin3(x))
        # x = self.act3(self.lin_gru(x))
        x = self.gru(x, hiddenState)
        hiddenState = x

        r = rawUnits
        r = torch.cat([self.embedding(r[:, :, 0].to(dtype=torch.long)), r[:, :, 1:]], dim=2)
        r = self.act_u1(self.lin_u1(r))
        r = self.act_u2(self.lin_u2(r) + self.lin_g2u(x).view(batch_size, 1, -1) + self.lin_c2u(minimapUnitInfo))
        r = self.act_u2_2(self.lin_u2_2(r))
        r = self.act_u2_3(self.lin_u2_3(r))
        r = self.act_u3(self.lin_u3(r))

        # r = torch.zeros((batch_size, num_units, 16)).cuda()
        # r = self.act_u3(self.lin_u3(r))

        return r, hiddenState


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

module_name = "movement"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
small_input = False

print("Loading")
data_paths = ["training_data/replays/s2"]
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
trainer = TrainerRNN(model, optimizer, action_loss_weights=[1, 1], device=device, sampleClass=MovementTrace, stepperClass=MovementStepper, step_size=1)
learning_rate_decay = 200

padding = PadSequence(MovementTrace(
    states='stack-timewise',
    movement='pad-timewise',
    replay_path=None,
    minimap_states='stack-timewise',
    raw_unit_states='pad-timewise',
    data_path=None,
    playerID=None,
    raw_unit_coords='pad-timewise'
))


def visualize(epoch):
    import matplotlib
    # Fix crash bug on some macOS versions
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    # Turn on non-blocking plotting mode
    plt.ion()

    load_weights(f"models/{module_name}_{epoch}.weights")
    memory, test_memory = create_datasets(cache_dir, test_split)
    for sampleIndex in range(1000):
        sample = memory[sampleIndex]
        with gzip.open(sample.data_path, "rb") as f:
            session = pickle.load(f)
        stepper = trainer.stepperClass(model, MovementTrace(*[[x] for x in sample]), device, 0, lambda a, b: 0, step_size=1)
        map_size = game_state_loader.find_map_size(session)
        while True:
            timestep = stepper.timestep
            res = stepper.step()
            if res is None:
                break

            outputs = stepper.outputs.detach().cpu().numpy()
            playerIndex = sample.playerID - 1
            units = session["observations"][playerIndex]["rawUnits"][timestep]["units"]
            our_units = [u for u in units if u["owner"] == sample.playerID and game_state_loader.isMovableUnit(u, buildOrderLoader)]
            our_buildings = [u for u in units if u["owner"] == sample.playerID and not game_state_loader.isMovableUnit(u, buildOrderLoader)]

            assert outputs.shape == (1, len(our_units), 2)
            mirror = sample.playerID == 2
            coords = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in our_units])
            coords2 = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in our_buildings])

            fig, axs = plt.subplots(nrows=3, ncols=7, figsize=(16, 6), num=1, clear=True)

            plt.sca(axs[0, 0])
            edgecolors = [(0, 0, 0, 1) if unit_lookup.military_units_mask[unit_lookup.unit_index_map[u["unit_type"]]] else (0, 0, 0, 0) for u in our_units]

            enemy_units = [u for u in units if u["owner"] != sample.playerID and u["unit_type"] in unit_lookup.unit_index_map]
            enemy_coords = np.array([game_state_loader.transform_coord(u["pos"], map_size, mirror) for u in enemy_units]).reshape(-1, 2)
            plt.scatter(enemy_coords[:, 0], enemy_coords[:, 1], c="#e41a1c")
            plt.scatter(coords2[:, 0], coords2[:, 1], c="#999999")
            plt.scatter(coords[:, 0], coords[:, 1], c=np.exp(outputs[0, :, 1]), cmap=plt.cm.viridis, edgecolors=edgecolors)

            plt.xlim((-0.1, 1.1))
            plt.ylim((-0.1, 1.1))
            # plt.colorbar()
            plt.clim((0, 1))
            axs[0, 0].set(aspect='equal')

            plt.sca(axs[1, 0])
            plt.scatter(enemy_coords[:, 0], enemy_coords[:, 1], c="#e41a1c")
            plt.scatter(coords[:, 0], coords[:, 1], c=sample.movement[timestep], cmap=plt.cm.viridis)

            plt.xlim((-0.1, 1.1))
            plt.ylim((-0.1, 1.1))
            # plt.colorbar()
            plt.clim((0, 1))
            axs[1, 0].set(aspect='equal')

            minimap = model.debug_last_minimap_output.detach().cpu()

            for i in range(model.minimap_layers // 2):
                for j in range(2):
                    plt.sca(axs[j, i + 1])
                    plt.imshow(sample.minimap_states[timestep][i + j * model.minimap_layers // 2].transpose(0, 1), origin="lower")

            for i in range(6):
                plt.sca(axs[2, i])
                plt.imshow(minimap[0, i].transpose(0, 1), origin="lower")

            # plt.tight_layout()
            plt.pause(0.01)
            # stepper.outputs


def cache_tensors():
    def load_state(s, target_filepath):
        def save_sample(sample):
            with gzip.open(target_filepath, 'wb') as f:
                torch.save(sample, f)
        game_state_loader.loadSessionMovement(s, buildOrderLoader, save_sample, None)

    version = 0
    common.cache_tenors(data_paths, cache_dir, small_input, load_state, version)


def learning_rate_by_time(epoch):
    return learning_rate - (learning_rate - 0.0001) * min(epoch / learning_rate_decay, 1)


assert learning_rate_by_time(0) == learning_rate
assert abs(learning_rate_by_time(learning_rate_decay) - 0.0001) < 0.00001
assert abs(learning_rate_by_time(learning_rate_decay + 100) - 0.0001) < 0.00001


def plot(tensorboard_writer):
    tensorboard_writer.add_embedding(model.embedding.weight, metadata=[u[0] for u in unit_lookup.units], global_step=current_step, tag="unit embedding")


def save_tensorboard_graph(memory, tensorboard_writer):
    sample = memory[0]
    # tensorboard_writer.add_graph(model, torch.tensor([sample.state], dtype=torch.float, device=device), True)


def train(comment):
    from tensorboardX import SummaryWriter
    print(f"Parameters: {count_parameters(model)}")

    global training_losses_by_time, current_step
    tensorboard_writer = SummaryWriter(log_dir=f"tensorboard/{module_name}/{datetime.now():%Y-%m-%d_%H:%M} {comment}")
    memory, test_memory = create_datasets(cache_dir, test_split)
    save_tensorboard_graph(test_memory, tensorboard_writer)

    training_generator = torch.utils.data.DataLoader(memory, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=padding)
    testing_generator = torch.utils.data.DataLoader(test_memory, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2, collate_fn=padding)

    for epoch, current_step in training_loop(training_generator, testing_generator, trainer, tensorboard_writer):
        for g in optimizer.param_groups:
            g['lr'] = learning_rate_by_time(epoch)
            print("Set learning rate to ", g['lr'])

        if epoch > 1:
            save(epoch)
            plot(tensorboard_writer)


def save(epoch):
    torch.save(model.state_dict(), f"models/{module_name}_" + str(epoch) + ".weights")


def load_weights(file):
    model.load_state_dict(torch.load(file, map_location='cpu'))


class Stepper:
    def __init__(self):
        self.stepper = trainer.stepperClass(model, None, device, 0, lambda a, b: 0, step_size=1)
        self.stepper.init_hidden_states(batch_size=1)

    def step(self, json_data, playerID):
        observer_session = json.loads(json_data)
        trace = game_state_loader.loadSessionMovement2(observer_session, playerID, buildOrderLoader, False, "invalid")
        self.stepper.set_batch(padding([trace])[0])
        self.stepper.step()
        result = self.stepper.outputs.detach().cpu().exp().numpy()[0, :, 1].tolist()
        print(result)
        return result


if __name__ == "__main__":
    common.train_interface(cache_tensors, train, visualize)
