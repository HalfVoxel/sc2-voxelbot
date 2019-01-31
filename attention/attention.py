import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
import math
# Fix crash bug on some macOS versions
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionEncoder(nn.Module):
    def __init__(self, max_sequence_length, input_size, key_size=4):
        super(AttentionEncoder, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.input_size = input_size
        self.key_size = key_size

        # Input x Mkey, Input x Mval, Input x Mquery
        self.key = nn.Linear(input_size, key_size)
        self.query = nn.Linear(input_size, key_size)
        self.value_size = input_size
        self.value = nn.Linear(input_size, self.value_size)
        self.bias = nn.Linear(input_size, 1)

        # self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.lin1 = nn.Linear(self.value_size, self.input_size)
        self.act1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(self.input_size)
        self.norm2 = nn.LayerNorm(self.input_size, elementwise_affine=False)

    def forward(self, input):
        batch_size = input.size()[0]
        assert input.size()[1:] == (self.max_sequence_length, self.input_size)
        # assert input_mask.size() == (batch_size,)

        keys = self.key(input)
        keys *= math.sqrt(1.0 / self.key_size)
        queries = self.query(input)
        values = self.value(input)
        biases = self.bias(input)
        # input_mask = input_mask.view((batch_size, 1))

        # Note: the 1 at the end is important for broadcasting
        assert biases.size() == (batch_size, self.max_sequence_length, 1)

        # Calculate all dot products between the queries and values
        # bmm = batch matrix multiplication
        scores = torch.bmm(keys, torch.transpose(queries, 1, 2)) + biases
        assert scores.size() == (batch_size, self.max_sequence_length, self.max_sequence_length)

        # Apply a non-linearity
        scores = scores.sigmoid()
        # print(scores)

        # (BxSxS) x (BxSxN) = (BxSxN)
        attention = torch.bmm(scores, values)
        # Residual normalization 1
        res = self.norm1(input + attention)
        # res = attention
        feedforward = self.act1(self.lin1(res))
        # Residual normalization 2
        result = self.norm2(res + feedforward)
        # result = feedforward
        return result


class PointModel(nn.Module):
    def __init__(self, max_sequence_length, input_size, heads, key_size=4):
        super(PointModel, self).__init__()
        self.input_size = input_size
        self.heads = heads
        self.key_size = key_size
        self.max_sequence_length = max_sequence_length

        self.lin0 = nn.Linear(self.input_size, self.input_size)
        self.act0 = nn.ReLU()

        self.encoder = AttentionEncoder(max_sequence_length, input_size, key_size)
        self.encoder2 = AttentionEncoder(max_sequence_length, input_size, key_size)
        self.encoder3 = AttentionEncoder(max_sequence_length, input_size, key_size)
        self.encoder4 = AttentionEncoder(max_sequence_length, input_size, key_size)
        # self.encoder5 = AttentionEncoder(max_sequence_length, input_size, key_size)
        # self.encoder6 = AttentionEncoder(max_sequence_length, input_size, key_size)
        self.key = nn.Linear(input_size, key_size//self.heads)
        self.value_size = input_size//self.heads
        self.value = nn.Linear(input_size, self.value_size)
        self.queries = torch.nn.Parameter(data=torch.randn((self.heads, key_size//self.heads), device=device, dtype=torch.float), requires_grad=True)
        # self.queries2 = torch.nn.Parameter(data=torch.randn((input_size), device=device, dtype=torch.float), requires_grad=True)
        self.softmax = nn.Softmax(dim=2)
        self.lin1 = nn.Linear(self.input_size, 2)

    def forward(self, input):
        batch_size = input.size()[0]
        encoded = self.act0(self.lin0(input))
        encoded = self.encoder(input)
        encoded = self.encoder2(encoded)
        encoded = self.encoder3(encoded)
        encoded = self.encoder4(encoded)
        # encoded = self.encoder5(encoded)
        # encoded = self.encoder6(encoded)
        keys = self.key(encoded)
        keys *= math.sqrt(1.0 / self.key_size)

        values = self.value(encoded)  # BxSxN
        # print(keys)
        # print(encoded)
        # BxSxK x KxH = BxSxH
        scores = torch.matmul(keys, torch.transpose(self.queries, 0, 1))
        assert scores.size() == (batch_size, self.max_sequence_length, self.heads)

        # Apply a non-linearity
        scores = torch.transpose(scores, 1, 2)
        scores = self.softmax(scores)
        scores = torch.transpose(scores, 1, 2)
        # print(scores)
        # print(scores)

        # print(scores)

        # BxHxS x BxSXV = BxHxV
        attention = torch.bmm(torch.transpose(scores, 1, 2), values)
        assert attention.size() == (batch_size, self.heads, self.value_size)

        attention = attention.view(batch_size, -1)
        res = self.lin1(attention)
        res = res.view((-1, 2))
        # res += self.queries2
        return res


class PointModel2(nn.Module):
    def __init__(self, max_sequence_length, input_size, key_size=4):
        super(PointModel2, self).__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.max_sequence_length = max_sequence_length

        self.lin0 = nn.Linear(self.input_size, self.input_size)
        self.act0 = nn.LeakyReLU()

        self.lin1 = nn.Linear(max_sequence_length * self.input_size, 16)

        self.lin2 = nn.Linear(16, 16)
        self.lin3 = nn.Linear(16, 8)
        self.lin4 = nn.Linear(8, 2)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        batch_size = input.size()[0]
        x = self.act0(self.lin0(input))
        x = x.view((batch_size, -1))
        x = self.act0(self.lin1(x))
        x = self.act0(self.lin2(x))
        x = self.act0(self.lin3(x))
        x = self.act0(self.lin4(x))
        return x


def generate_dataset(size):
    # points = np.random.normal((0,0), (1,1), (size, 2))
    # result = np.stack([np.cos(points[:,0]), np.sin(points[:,0]), np.cos(points[:,1]), np.sin(points[:,1])], axis=1)
    points = np.random.uniform(0, 1, (size, 2))
    pi = math.pi
    result = np.stack([
        # np.cos(0.5 * points[:, 0] * pi), np.sin(0.5 * points[:, 0] * pi),
        np.cos(1.0 * points[:, 0] * pi), np.sin(1.0 * points[:, 0] * pi),
        np.cos(2.0 * points[:, 0] * pi), np.sin(2.0 * points[:, 0] * pi),
        np.cos(4.0 * points[:, 0] * pi), np.sin(4.0 * points[:, 0] * pi),
        # np.cos(0.5 * points[:, 1] * pi), np.sin(0.5 * points[:, 1] * pi),
        np.cos(1.0 * points[:, 1] * pi), np.sin(1.0 * points[:, 1] * pi),
        np.cos(2.0 * points[:, 1] * pi), np.sin(2.0 * points[:, 1] * pi),
        np.cos(4.0 * points[:, 1] * pi), np.sin(4.0 * points[:, 1] * pi),
    ], axis=1)

    threshold = 0.1
    # threshold = 0.05
    close = False
    close2 = False

    dists = np.zeros(size)

    # Check pairwise distances
    for i in range(size - 1):
        dists[i] = np.sqrt(np.min(((points[i, :] - points[i + 1:, :])**2).sum(axis=1)))
        if np.any(((points[i, :] - points[i + 1:, :])**2).sum(axis=1) < threshold * threshold):
            assert dists[i] < threshold
            close = True

    # dists[-1] = 1
    # result[:,2] = dists

    # assert (np.min(dists) < threshold) == close, (dists, threshold, close)

    # idx = random.randrange(0,size)
    # idx2 = (idx + 1) % size
    # result[:,1] = 0
    # result[idx,1] = 1
    # result[idx2,1] = 1
    # result[idx,0] = 0
    # result[idx2,0] = 1 if close else 0

    # close = True
    return (result, close)


torch.set_printoptions(precision=2)
values = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13], [14, 15, 16], [17, 18, 19]]], dtype=torch.float)
queries = torch.tensor([[0.5, 0.5, 0]], dtype=torch.float)
keys = torch.matmul(values, torch.transpose(queries, 0, 1))
# print(keys)
scores = torch.tensor([[[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 1]]], dtype=torch.float)
# attention = torch.matmul(torch.transpose(scores, 1, 2), values)
attention = torch.bmm(scores, values)
print(attention)
# exit(0)
# input = torch.randn(2, 2, 3, 3)
# input[0,0,0,:] = torch.tensor([1, 1, 1])
# input[0,0,1,:] = torch.tensor([0, 0, 0])
# input[0,0,2,:] = torch.tensor([0, 1, 2])
# print(input)
# # With Learnable Parameters
# m = nn.LayerNorm(3)
# print(m(input))
# # Without Learnable Parameters
# m = nn.LayerNorm(3, elementwise_affine=False)
# print(m(input))
# # Normalize over last two dimensions
# m = nn.LayerNorm([3, 3])
# print(m(input))
# # Normalize over last dimension of size 10
# m = nn.LayerNorm(3)
# # Activating the module
# print(m(input))

num_points = 10
train_data = []
test_data = []
BATCH_SIZE = 64
model = PointModel(max_sequence_length=num_points, heads=2, input_size=12, key_size=8)
episode_index = 0
num_samples = 20000
losses = []
test_losses = []
print("Generating training data")
print(list(model.named_parameters()))
for i in range(num_samples):
    train_data.append(generate_dataset(num_points))

print("Generating test data")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Parameters: {count_parameters(model)}")
# exit(0)
for i in range(num_samples // 10):
    test_data.append(generate_dataset(num_points))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

target = [float(x[1]) for x in train_data]
print(sum(target), len(target))
# exit(0)


def plot():
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    ls = np.array(losses)
    plt.plot(ls[:, 0], ls[:, 1])
    ls = np.array(test_losses)
    plt.plot(ls[:, 0], ls[:, 1])

    plt.pause(0.001)  # pause a bit so that plots are updated


def evaluate_batch(transitions):
    input = torch.tensor(np.stack([x[0] for x in transitions]), dtype=torch.float)
    target = torch.tensor(np.stack([float(x[1]) for x in transitions]), dtype=torch.long)
    output = model(input)
    loss = loss_fn(output, target)
    return loss


def optimize_model():
    global episode_index

    steps = len(train_data) // BATCH_SIZE
    for i in range(steps):
        model.train()
        transitions = random.choices(train_data, k=BATCH_SIZE)
        loss = evaluate_batch(transitions)

        print(loss.item())
        losses.append([episode_index + (i / steps), loss.item()])

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            if name in ["queries"] and False:
                print(param.data)
            # param.grad.data.clamp_(-1, 1)
        optimizer.step()

    episode_index += 1

    with torch.no_grad():
        model.eval()
        test_loss = evaluate_batch(test_data)
        print(f"Test loss: {test_loss.item()}")
        # print(model.queries)
        test_losses.append([episode_index, test_loss.item()])

    plot()


while True:
    optimize_model()
