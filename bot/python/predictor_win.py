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
import inspect
from collections import defaultdict
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
vis = visdom.Visdom()
tensorboard_writer = SummaryWriter()

class AttentionNet2(nn.Module):
    def __init__(self, input_shape, num_input_units, num_units):
        super().__init__()

        layers = []
        self.num_units = num_units
        self.embedding = nn.Embedding(num_units, 8)

        self.decoder1 = AttentionDecoder(num_input_units, input_size=11 + (self.embedding.embedding_dim - 1), key_size=4, value_size=2, heads=8, output_sequence_length=20)
        self.dlin1 = nn.Linear(self.decoder1.output_size, self.decoder1.output_size)
        self.self_attention1 = SelfAttention(max_sequence_length=self.decoder1.output_sequence_length, input_size=self.decoder1.output_size, key_size=4, heads=1)
        self.decoder2 = AttentionDecoder(max_sequence_length=self.decoder1.output_sequence_length, input_size=self.decoder1.output_size, key_size=4, value_size=8, heads=4, output_sequence_length=20)
        self.dlin2 = nn.Linear(self.decoder2.output_size, self.decoder2.output_size)
        self.self_attention2 = SelfAttention(max_sequence_length=self.decoder2.output_sequence_length, input_size=self.decoder2.output_size, key_size=8, heads=1)
        self.decoder3 = AttentionDecoder(max_sequence_length=self.decoder2.output_sequence_length, input_size=self.decoder2.output_size, key_size=8, value_size=16, heads=2, output_sequence_length=1)
        self.dlin3 = nn.Linear(self.decoder3.output_size, self.decoder3.output_size)
        # self.self_attention3 = SelfAttention(max_sequence_length=self.decoder3.output_sequence_length, input_size=self.decoder3.output_size, key_size=8, heads=1)
        # self.decoder4 = AttentionDecoder(max_sequence_length=self.decoder3.output_sequence_length, input_size=self.decoder3.output_size, key_size=8, value_size=16, heads=2, output_sequence_length=1)
        # self.dlin4 = nn.Linear(self.decoder4.output_size, self.decoder4.output_size)
        
        self.lin1 = nn.Linear(self.decoder3.output_size, 32)
        self.linK = nn.Linear(11 + (self.embedding.embedding_dim - 1), 32)
        self.act1 = nn.LeakyReLU()

        self.gru_hidden_size = 32
        self.gru = nn.GRUCell(input_size=32, hidden_size=self.gru_hidden_size)
        self.lin2 = nn.Linear(32, 32)
        self.act = nn.LeakyReLU()

        self.lin3 = nn.Linear(32, 32)

        self.lin4 = nn.Linear(32, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden_states(self, batch_size, device):
        return torch.zeros((batch_size, self.gru_hidden_size), device=device)

    def forward(self, inputTensor, rawUnitTensor, mask, stateTensor):
        batch_size = inputTensor.size()[0]

        embeddedUnitTypes = self.embedding(rawUnitTensor[:, :, 0].to(dtype=torch.long))
        rawUnitTensor = torch.cat([rawUnitTensor[:, :, 1:], embeddedUnitTypes], dim=2)

        x = rawUnitTensor
        # x = self.decoder1(x, mask)
        # x = self.act1(self.dlin1(x))
        # x = self.self_attention1(x, None)
        # x = self.decoder2(x, None)
        # x = self.act1(self.dlin2(x))
        # x = self.self_attention2(x, None)
        # x = self.decoder3(x, None)
        # x = self.act1(self.dlin3(x))
        x = x.sum(dim=1)
        x = self.act1(self.linK(x))

        # x = self.self_attention3(x, None)
        # x = self.decoder4(x, None)
        # x = self.act1(self.dlin4(x))
        x = x.view(batch_size, -1)
        x = self.act1(self.lin1(x))

        stateTensor = self.gru(x, stateTensor)
        x = stateTensor
        x = self.act(self.lin2(x))
        x = self.act(self.lin3(x))
        x = self.lin4(x)
        x = self.softmax(x)
        return x, stateTensor

# @torch.jit.trace
class AttentionNet(nn.Module):
    def __init__(self, input_shape, num_input_units, num_units):
        super().__init__()

        layers = []
        self.num_units = num_units
        self.embedding = nn.Embedding(num_units, 8)
        self.attention = AttentionModule(max_sequence_length=num_input_units, heads=6,
                                         input_size=11 + (self.embedding.embedding_dim - 1), key_size=4, value_size=4, self_attention_key_size=4, output_size=64, self_attention_heads=3)

        self.lin0 = nn.Linear(self.attention.output_size + 32, 64)
        self.act0 = nn.LeakyReLU()
        # layers.append(nn.BatchNorm1d(64))
        # layers.append(nn.Linear(128, 32))
        # layers.append(nn.Tanh())

        self.gru_hidden_size = 32
        self.gru = nn.GRUCell(input_size=64, hidden_size=self.gru_hidden_size)
        self.lin1 = nn.Linear(32, 32)
        self.act1 = nn.LeakyReLU()

        self.lin2 = nn.Linear(32, 32)
        self.act2 = nn.LeakyReLU()

        # layers.append(nn.BatchNorm1d(128))
        # layers.append(nn.Linear(128, 128))
        # layers.append(nn.Tanh())

        # layers.append(nn.BatchNorm1d(64))
        self.lin3 = nn.Linear(32, 2)
        self.act3 = nn.LeakyReLU()
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

        self.lin4 = nn.Linear(input_shape[-1], 16)
        self.act4 = nn.LeakyReLU()
        self.lin5 = nn.Linear(2 * 16, 32)
        self.act5 = nn.LeakyReLU()

        self.lin_gru = nn.Linear(64, self.gru_hidden_size)
        self.act_gru = nn.LeakyReLU()

    def init_hidden_states(self, batch_size, device):
        return torch.zeros((batch_size, self.gru_hidden_size), device=device)

    def forward(self, inputTensor, rawUnitTensor, mask, stateTensor):
        batch_size = inputTensor.size()[0]

        x = self.act4(self.lin4(inputTensor))
        x = x.view(batch_size, -1)
        x = self.act5(self.lin5(x))
        # embeddedUnitTypes = self.embedding(rawUnitTensor[:, :, 0].to(dtype=torch.long))
        # rawUnitTensor = torch.cat([rawUnitTensor[:, :, 1:], embeddedUnitTypes], dim=2)
        # attention = self.attention(rawUnitTensor, mask)
        attention = torch.zeros((batch_size, 64), device=device)
        # x = torch.zeros_like(x)
        x = torch.cat([x, attention], dim=1)

        x = self.act0(self.lin0(x))
        # stateTensor = self.gru(x, stateTensor)
        # x = stateTensor
        x = self.act_gru(self.lin_gru(x))

        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.lin3(x)
        x = self.softmax(x)
        return x, stateTensor


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
        # stateTensors = self.gru(x, stateTensors)
        # x = stateTensors
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
            game_state_loader.loadSession(s, buildOrderLoader, test_memory if random.uniform(
                0, 1) < test_split else memory, statistics)
        print("Done")

    with gzip.open(caching_filename, 'wb') as f:
        batches = list(split_into_batches(memory.get_all(), 64))
        pickle.dump(len(batches), f, protocol=pickle.HIGHEST_PROTOCOL)
        for chunk in batches:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_memory.get_all(), f, protocol=pickle.HIGHEST_PROTOCOL)


frame = inspect.currentframe()          # define a frame to track
# gpu_tracker = MemTracker(frame)         # define a GPU tracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
small_input = False
load_cached_data = True
caching_filename = "cached_tensors.pickle"
if small_input:
    caching_filename = "cached_tensors_small.pickle"

print("Loading")
data_paths = ["training_data/replays/7", "training_data/replays/8", "training_data/replays/9"]
# unit_lookup = UnitLookup(terranUnits + zergUnits + protossUnits)
unit_lookup = UnitLookup(protossUnits)
buildOrderLoader = BuildOrderLoader(unit_lookup, 1.0)

# gpu_tracker.track()

tensor_input_size = game_state_loader.getInputTensorSize(buildOrderLoader)
# model = Net((2, tensor_input_size), num_units=unit_lookup.num_units)
model = AttentionNet(input_shape=(2, tensor_input_size), num_input_units=200, num_units=unit_lookup.num_units)
model = model.to(device)

learning_rate = 0.004
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
test_split = 0.1
memory = ReplayMemory(5000000, prioritized_replay=False)
test_memory = ReplayMemory(5000000, prioritized_replay=False)
statistics = Statistics()
trainer = None
batch_size = 128
current_step = 0
# gpu_tracker.track()

training_losses = []
training_losses_by_time = []
test_losses = []
test_losses_by_time = []

def learning_rate_by_time(epoch):
    return learning_rate - (learning_rate - 0.0001) * min(epoch/20, 1)

assert learning_rate_by_time(0) == learning_rate
assert abs(learning_rate_by_time(20) - 0.0001) < 0.00001
assert abs(learning_rate_by_time(200) - 0.0001) < 0.00001


def count_parameters(model):
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Parameters: {count_parameters(model)}")

def split_into_batches(l, batch_size):
    """
    :param l:           list
    :param batch_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), batch_size):
        yield l[i:min(len(l), i + batch_size)]


plot_windows = defaultdict(lambda: None)

def plot():
    fig = plt.figure(1)
    plt.clf()
    # ax = fig.add_subplot(2, 2, 1)
    plt.title(' Training/Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid()
    losses = np.array(training_losses)
    plt.plot(losses[:, 0], losses[:, 1])
    # plot_windows["loss2"] = vis.line(X=losses[:, 0], Y=losses[:, 1], win=plot_windows["loss2"], name="trace1")
    losses = np.array(test_losses)
    plt.plot(losses[:, 0], losses[:, 1])
    # plot_windows["loss2"] = vis.line(X=losses[:, 0], Y=losses[:, 1], win=plot_windows["loss2"], update="append", name="trace2")
    # plot_windows["loss"] = vis.matplot(plt, win=plot_windows["loss"])
    tensorboard_writer.add_figure("loss", fig, global_step=current_step, close=True)


    fig = plt.figure(2)
    plt.clf()
    for i in range(2):
        ax = fig.add_subplot(1, 2, 1+i)
        source = test_losses_by_time[-1] if i == 0 else training_losses_by_time
        losses = np.array([(x[0], x[1]) for x in source])
        xs = losses[:, 0]
        probs = np.exp(-losses[:, 1])

        bins = 10
        means = np.zeros(bins)
        means_xs = np.zeros(bins)
        for bin_index in range(bins):
            means_xs[bin_index] = (bin_index + 0.5) / bins
            means[bin_index] = np.mean(probs[np.where((xs >= bin_index / bins) & (xs <= (bin_index + 1) / bins))])

        worst_index = np.argmin(probs * ((1 - xs) + 0.2))
        worst_trace = source[worst_index][2]
        mask_values = [1 if (x[2].replay_path == worst_trace.replay_path) else 0 for x in source]
        mask = np.array(mask_values)
        mask_indices = np.where(mask)
        print(f"Worst replay: {worst_trace.replay_path}")

        plt.title("Correct outcome (test)" if i == 0 else "Correct outcome (train)")
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.grid()
        plt.ylim([0, 1])
        plt.xlim([0, 1])

        plt.scatter(xs, probs, marker='.', alpha=0.3 if i == 0 else 0.2, s=1, color="#377eb8")
        plt.scatter(losses[mask_indices, 0], probs[mask_indices], marker='.', alpha=0.5, s=1, color="#e41a1c")
        plt.plot(means_xs, means, color="#000000")
        # plot_windows[f"prob_{i}"] = vis.matplot(plt, win=plot_windows[f"prob_{i}"])

    tensorboard_writer.add_figure("Probabilities", fig, global_step=current_step, close=True)

    # plt.tight_layout()
    # plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()
    print(model.embedding.weight)
    tensorboard_writer.add_embedding(model.embedding.weight, metadata=[u[0] for u in unit_lookup.units], global_step=current_step, tag="unit embedding")


def load_stuff():
    global trainer
    if load_cached_data:
        load_cached()
    else:
        load_all(0)


    sample = test_memory.sample(1)[0]
    tensorboard_writer.add_graph(model, (torch.tensor([sample.states[0]], dtype=torch.float, device=device), torch.tensor([sample.raw_unit_states[0]], dtype=torch.float, device=device), torch.tensor([sample.masks[0]], dtype=torch.uint8, device=device), model.init_hidden_states(1, device)), True)
    # trainer = PredictBuildOrder(model, optimizer, action_loss_weights=action_loss_weights)
    trainer = TrainerWinPredictor(model, optimizer, action_loss_weights=None, device=device)


def train():
    global training_losses_by_time, current_step
    load_stuff()
    step = 0
    epoch = 0
    while True:
        for g in optimizer.param_groups:
            g['lr'] = learning_rate_by_time(epoch)
            print("Set learning rate to ", g['lr'])

        epoch += 1
        trainer.time_threshold = math.exp(-epoch / 5.0)
        training_losses_by_time = []
        all_samples = memory.get_all()
        random.shuffle(all_samples)
        batches = list(split_into_batches(all_samples, batch_size))
        print(f"\rTraining Epoch {epoch}", end="")
        for i in range(len(batches)):
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            loss, losses_by_time = trainer.train(batches[i])
            print(f"\rTraining Epoch {epoch} [{i+1}/{len(batches)}] loss={loss}", end="")
            training_losses.append((step, loss))
            step += 1
            current_step = step
            training_losses_by_time += losses_by_time
            tensorboard_writer.add_scalar("training loss", loss, step)

            # print(prof.key_averages().table(sort_by="cpu_time_total"))

        print()
        if len(test_memory) > 0:
            trainer.time_threshold = 0
            all_samples = test_memory.get_all()
            batches = list(split_into_batches(all_samples, batch_size))
            total_loss = 0
            total_losses_by_time = []
            for i in range(len(batches)):
                loss, losses_by_time = trainer.test(batches[i])
                total_losses_by_time += losses_by_time
                total_loss += loss * len(batches[i])

            total_loss /= len(all_samples)
            test_losses_by_time.append(total_losses_by_time)

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
