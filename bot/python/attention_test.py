import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib
import math
from attention import AttentionModule

# Fix crash bug on some macOS versions
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class ConvModel(nn.Module):
    def __init__(self, input_size):
        super(ConvModel, self).__init__()
        self.input_size = input_size

        kernel = 3
        self.conv00 = nn.Conv2d(in_channels=2 + 20, out_channels=2 + 10,
                                kernel_size=kernel, stride=1, padding=kernel // 2)
        self.pool00 = nn.MaxPool2d(2, stride=None, padding=0)
        self.act00 = nn.LeakyReLU()

        self.conv0 = nn.Conv2d(in_channels=2 + 10, out_channels=4, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.pool0 = nn.MaxPool2d(2, stride=None, padding=0)
        self.act0 = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.pool1 = nn.MaxPool2d(2, stride=None, padding=0)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel, stride=1, padding=kernel // 2)
        self.pool2 = nn.MaxPool2d(5, stride=None, padding=0)
        self.act2 = nn.LeakyReLU()

        self.lin1 = nn.Linear(16, 2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        batch_size = input.size()[0]
        x = input
        x = self.act00(self.conv00(x))
        print(x.size())
        assert x.size() == (batch_size, 2 + 10, 40, 40)
        x = self.pool00(x)
        assert x.size() == (batch_size, 2 + 10, 20, 20)

        x = self.act0(self.conv0(x))
        assert x.size() == (batch_size, 4, 20, 20)
        x = self.pool0(x)
        assert x.size() == (batch_size, 4, 10, 10)

        x = self.act1(self.conv1(x))
        assert x.size() == (batch_size, 8, 10, 10)
        x = self.pool1(x)
        assert x.size() == (batch_size, 8, 5, 5)

        x = self.act2(self.conv2(x))
        assert x.size() == (batch_size, 16, 5, 5)
        x = self.pool2(x)
        assert x.size() == (batch_size, 16, 1, 1)

        x = x.view(batch_size, 16)
        x = self.act0(self.lin1(x))
        assert x.size() == (batch_size, 2)
        return x


def generate_dataset(size):
    # points = np.random.normal((0,0), (1,1), (size, 2))
    # result = np.stack([np.cos(points[:,0]), np.sin(points[:,0]), np.cos(points[:,1]), np.sin(points[:,1])], axis=1)
    clusters = 4
    points = np.random.uniform(0, 1, (clusters, 2))
    labels = np.random.randint(0, 2, size=(clusters,))
    points = np.tile(points, (size // clusters, 1))
    labels = np.tile(labels, size // clusters)
    points = points + np.random.normal((0, 0), (0.05, 0.05), (size, 2))

    pi = math.pi
    result = np.concatenate([
        labels.reshape(size, 1),
        # np.cos(0.5 * points[:, 0] * pi), np.sin(0.5 * points[:, 0] * pi),
        np.cos(1.0 * points[:, 0] * pi).reshape(size, 1), np.sin(1.0 * points[:, 0] * pi).reshape(size, 1),
        np.cos(2.0 * points[:, 0] * pi).reshape(size, 1), np.sin(2.0 * points[:, 0] * pi).reshape(size, 1),
        np.cos(4.0 * points[:, 0] * pi).reshape(size, 1), np.sin(4.0 * points[:, 0] * pi).reshape(size, 1),
        # np.cos(0.5 * points[:, 1] * pi).reshape(size, 1), np.sin(0.5 * points[:, 1] * pi),
        np.cos(1.0 * points[:, 1] * pi).reshape(size, 1), np.sin(1.0 * points[:, 1] * pi).reshape(size, 1),
        np.cos(2.0 * points[:, 1] * pi).reshape(size, 1), np.sin(2.0 * points[:, 1] * pi).reshape(size, 1),
        np.cos(4.0 * points[:, 1] * pi).reshape(size, 1), np.sin(4.0 * points[:, 1] * pi).reshape(size, 1),
        np.ones(shape=(size, 20)),
    ], axis=1)

    mask = np.ones(size)
    maskIndex = random.randrange(1, size)
    mask[maskIndex:] = 0

    threshold = 0.1
    # threshold = 0.05
    close = False

    dists = np.zeros(size)

    # Check pairwise distances
    for i in range(maskIndex - 1):
        indices = np.where(labels[:maskIndex] != labels[i])[0]
        # print(indices)
        if len(indices) > 0:
            dists[i] = np.sqrt(np.min(((points[i, :] - points[indices, :])**2).sum(axis=1)))
            # print(dists[i])
            if dists[i] < threshold:
                close = True

    d = 40
    result2D = np.zeros((2 + 20, d, d), dtype=np.float)
    result2D = np.random.normal(size=(2 + 20, d, d))
    result2D[0:2, :, :] = 0
    result2D[2:, :, :] = 1
    for i in range(size):
        x = max(0, min(d - 1, int(points[i, 0] * d)))
        y = max(0, min(d - 1, int(points[i, 1] * d)))
        result2D[labels[i], x, y] = 1

    result = torch.tensor(result, dtype=torch.float)
    maskTensor = torch.tensor(mask, dtype=torch.uint8)
    # colors = ['red','green']
    # plt.scatter(points[:,0], points[:,1], c=labels)
    # print("Close: ", close)
    # plt.show()

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
    return (result, close, maskTensor)


torch.set_printoptions(precision=2)
values = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 12, 13],
                                                           [14, 15, 16], [17, 18, 19]]], dtype=torch.float)
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

num_points = 100
train_data = []
test_data = []
BATCH_SIZE = 64
model = AttentionModule(max_sequence_length=num_points, heads=2, input_size=13 + 20, key_size=8, output_size=2, value_size=4)
#model = ConvModel(input_size=5)
model = model.to(device)
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
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ls = np.array(losses)
    plt.plot(ls[:, 0], ls[:, 1])
    ls = np.array(test_losses)
    plt.plot(ls[:, 0], ls[:, 1])

    plt.pause(0.001)  # pause a bit so that plots are updated


def evaluate_batch(transitions):
    input = torch.stack([x[0] for x in transitions]).to(device)
    target = torch.tensor([float(x[1]) for x in transitions], dtype=torch.long, device=device)
    mask = torch.stack([x[2] for x in transitions]).to(device)
    output = model(input, mask)
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


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    for i in range(1):
        optimize_model()

print(prof.key_averages())
