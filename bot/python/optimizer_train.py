import json
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data
from collections import namedtuple
import numpy as np
import math
import random
from scipy.stats import gaussian_kde as kde
import time
import mappings

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "training_data/buildorders_time/4"

manualSeed = 123
np.random.seed(manualSeed)
random.seed(manualSeed+1)
torch.manual_seed(manualSeed+2)

def addTrainingData(items):
    index = len(os.listdir(data_path))
    with open(data_path + "/chunk_" + str(index) + ".json", "w") as f:
        f.write(json.dump(items))


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)


units = set([
    "PROTOSS_ADEPT",
    "PROTOSS_ASSIMILATOR",
    "PROTOSS_CARRIER",
    "PROTOSS_COLOSSUS",
    "PROTOSS_CYBERNETICSCORE",
    "PROTOSS_DARKSHRINE",
    "PROTOSS_DARKTEMPLAR",
    "PROTOSS_DISRUPTOR",
    "PROTOSS_FLEETBEACON",
    "PROTOSS_FORGE",
    "PROTOSS_GATEWAY",
    "PROTOSS_HIGHTEMPLAR",
    "PROTOSS_IMMORTAL",
    "PROTOSS_MOTHERSHIP",
    "PROTOSS_NEXUS",
    "PROTOSS_OBSERVER",
    "PROTOSS_ORACLE",
    "PROTOSS_PHOENIX",
    "PROTOSS_PHOTONCANNON",
    "PROTOSS_PROBE",
    "PROTOSS_PYLON",
    "PROTOSS_ROBOTICSBAY",
    "PROTOSS_ROBOTICSFACILITY",
    "PROTOSS_SENTRY",
    "PROTOSS_SHIELDBATTERY",
    "PROTOSS_STALKER",
    "PROTOSS_STARGATE",
    "PROTOSS_TEMPEST",
    "PROTOSS_TEMPLARARCHIVE",
    "PROTOSS_TWILIGHTCOUNCIL",
    "PROTOSS_VOIDRAY",
    "PROTOSS_WARPGATE",
    "PROTOSS_WARPPRISM",
    "PROTOSS_ZEALOT",
])

economicallyRelevantUnitsProtoss = [
    "PROTOSS_ASSIMILATOR",
    "PROTOSS_CYBERNETICSCORE",
    "PROTOSS_DARKSHRINE",
    "PROTOSS_FLEETBEACON",
    "PROTOSS_FORGE",
    "PROTOSS_GATEWAY",
    "PROTOSS_NEXUS",
    "PROTOSS_PROBE",
    "PROTOSS_PYLON",
    "PROTOSS_ROBOTICSBAY",
    "PROTOSS_ROBOTICSFACILITY",
    "PROTOSS_STARGATE",
    "PROTOSS_TEMPLARARCHIVE",
    "PROTOSS_TWILIGHTCOUNCIL",
    "PROTOSS_WARPGATE",
]

ignoreCostUnits = [
    "PROTOSS_PROBE",
    "TERRAN_SCV",
    "ZERG_DRONE",
    "PROTOSS_NEXUS",
    "TERRAN_COMMANDCENTER",
    "TERRAN_ORBITALCOMMAND",
    "TERRAN_PLANETARYFORTRESS",
    "ZERG_HATCHERY",
    "ZERG_LAIR",
    "ZERG_HIVE",
    "PROTOSS_PYLON",
    "ZERG_OVERLORD",
    "TERRAN_SUPPLYDEPOT",

    "PROTOSS_ASSIMILATOR",
    "ZERG_EXTRACTOR",
    "TERRAN_REFINERY",
]

unitLookup = mappings.UnitLookup([u for u in mappings.protossUnits if u[0] in units] + mappings.protossUpgrades)
economicUnitLookup = mappings.UnitLookup([u for u in mappings.protossUnits if u[0] in economicallyRelevantUnitsProtoss] + [u for u in mappings.protossUpgrades if u[0] == "WARPGATERESEARCH"])
ignoreCostUnitsLookup = mappings.UnitLookup([u for u in mappings.allUnits if u[0] in ignoreCostUnits])
assert(len(ignoreCostUnitsLookup) == len(ignoreCostUnits))

NUM_UNITS = len(unitLookup.units)
NUM_UNITS_ECONOMICAL = len(economicUnitLookup.units)
STARTING_UNIT_TENSOR_SIZE = NUM_UNITS_ECONOMICAL * 3
TARGET_UNIT_TENSOR_SIZE = NUM_UNITS * 3
META_SIZE = 4
MAX_INSTANCE_TIME = 30 * 60

unitFoodRequirementTensor = torch.zeros(NUM_UNITS)
for i, u in enumerate(unitLookup.units):
    unitFoodRequirementTensor[i] = u.food_delta

class ListDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = []
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def add(self, item):
        self.data.append(item)

memory = ListDataset()
originalDatas = []
def load_instance(item):
    if item["buildOrderTime"] > MAX_INSTANCE_TIME:
        return
    
    if item["fitness"]["time"] == 100000:
        print("Skipping failed instance")
        return
    
    if "version" not in item:
        for v in [37, 38, 39, 49, 41, 42]:
            if v in item["buildOrder"]:
                # print("Skipping broken instance")
                return

    startingUnits = torch.zeros(NUM_UNITS_ECONOMICAL, dtype=torch.float)
    allStartingUnits = torch.zeros(NUM_UNITS, dtype=torch.float)
    targetUnits = torch.zeros(NUM_UNITS, dtype=torch.float)

    for u in item["startingUnits"]:
        allStartingUnits[unitLookup.unit_index_map[u["type"]]] += u["count"]

        if u["type"] in economicUnitLookup.unit_index_map:
            startingUnits[economicUnitLookup.unit_index_map[u["type"]]] += u["count"]

    for u in item["targetUnits"]:
        targetUnits[unitLookup.unit_index_map[u["type"]]] += u["count"]

    UPGRADE_OFFSET = 1000000
    for u in item["startingUpgrades"]:
        id = UPGRADE_OFFSET + u
        allStartingUnits[unitLookup.unit_index_map[id]] += 1

        if id in economicUnitLookup.unit_index_map:
            startingUnits[economicUnitLookup.unit_index_map[id]] += 1

    # print("Upgrades", item["targetUpgrades"])
    for u in item["targetUpgrades"]:
        # print(unitLookup[UPGRADE_OFFSET + u])
        # print(unitLookup.unit_index_map[UPGRADE_OFFSET + u])
        if u not in item["startingUpgrades"]:
            targetUnits[unitLookup.unit_index_map[UPGRADE_OFFSET + u]] += 1

    if allStartingUnits.sum() > 200:
        # print("Skipping item with", startingUnits.sum(), "units")
        return

    debug = False and len(item["targetUnits"]) == 0 and len(item["targetUpgrades"]) > 0

    if debug:
        print("Resources ", item["startingMinerals"], item["startingVespene"])
        for u in item["startingUnits"]:
            print("Start unit ", unitLookup[u["type"]].name, u["count"])
        for u in item["targetUnits"]:
            print("Target unit ", unitLookup[u["type"]].name, u["count"])
        print(item)
        print(targetUnits)
        print(allStartingUnits)
        print(startingUnits)
        exit(0)

    additionalMineralCost = 0
    additionalVespeneCost = 0
    for u in item["buildOrder"]:
        isTargetUnit = False
        for u2 in item["targetUnits"]:
            if u == u2["type"]:
                isTargetUnit = True
        
        if u >= UPGRADE_OFFSET:
            isTargetUnit = True

        # Only if unit is not nexus/probe & unit is not part of the target units
        if isTargetUnit:
            continue
        
        if u in ignoreCostUnitsLookup:
            continue
        
        unit = unitLookup[u]
        mineralCost = unit.mineral_cost
        vespeneCost = unit.vespene_cost

        additionalMineralCost += mineralCost
        additionalVespeneCost += vespeneCost
        if debug:
            print("Additional cost for", unit.name, mineralCost, vespeneCost)

    if debug:
        print(additionalMineralCost, additionalVespeneCost)

    # print(targetUnits)
    # print(startingFood)
    # targetUnits = torch.max(torch.tensor(0.0), targetUnits - allStartingUnits)
    targetFood = (targetUnits * unitFoodRequirementTensor).sum()
    startingFood = (allStartingUnits * unitFoodRequirementTensor).sum()

    startingUnits1 = (startingUnits > 0).to(dtype=torch.float)
    targetUnits1 = (targetUnits > 0).to(dtype=torch.float)
    startingUnits2 = (startingUnits == 1).to(dtype=torch.float)
    targetUnits2 = (targetUnits == 1).to(dtype=torch.float)

    startingUnits = torch.cat([startingUnits, startingUnits1, startingUnits2])
    targetUnits = torch.cat([targetUnits, targetUnits1, targetUnits2])

    meta = torch.zeros(META_SIZE)
    meta[0] = item["startingMinerals"] / 100
    meta[1] = item["startingVespene"] / 100
    meta[2] = startingFood / 10
    meta[3] = targetFood / 10

    # meta[2] = item["buildOrderTime"]

    assert meta.shape == (META_SIZE,)
    assert startingUnits.shape == (STARTING_UNIT_TENSOR_SIZE,)
    assert targetUnits.shape == (TARGET_UNIT_TENSOR_SIZE,)
    originalDatas.append(item)
    # memory.add((startingUnits, targetUnits, meta, item["buildOrderTime"], len(originalDatas) - 1, additionalMineralCost, additionalVespeneCost))
    memory.add((startingUnits, targetUnits, meta, item["fitness"]["time"], len(originalDatas) - 1, additionalMineralCost, additionalVespeneCost))

def predict(startingUnits, resources, targetUnitsList):
    startingUnitsTensor = torch.zeros(NUM_UNITS_ECONOMICAL, dtype=torch.float)
    allStartingUnitsTensor = torch.zeros(NUM_UNITS, dtype=torch.float)
    for u in startingUnits:
        if u[0] in unitLookup.unit_index_map:
            allStartingUnitsTensor[unitLookup.unit_index_map[u[0]]] += u[1]
        if u[0] in economicUnitLookup.unit_index_map:
            startingUnitsTensor[economicUnitLookup.unit_index_map[u[0]]] += u[1]
    
    startingFood = (allStartingUnitsTensor * unitFoodRequirementTensor).sum()
    startingUnits1 = (startingUnitsTensor > 0).to(dtype=torch.float)
    startingUnits2 = (startingUnitsTensor == 1).to(dtype=torch.float)

    startingUnitsTensor = torch.cat([startingUnitsTensor, startingUnits1, startingUnits2])
    targetUnitTensors = []
    metas = []
    offsets = []
    for targetUnits in targetUnitsList:
        targetUnitsTensor = torch.zeros(NUM_UNITS, dtype=torch.float)
        offset = 0
        for u in targetUnits:
            if u[0] == 141:
                # Convert archons to 2 high templars
                u = (75, u[1] * 2)

            targetUnitsTensor[unitLookup.unit_index_map[u[0]]] += u[1]

        targetFood = (targetUnitsTensor * unitFoodRequirementTensor).sum()
        targetUnits1 = (targetUnitsTensor > 0).to(dtype=torch.float)
        targetUnits2 = (targetUnitsTensor == 1).to(dtype=torch.float)
        targetUnitsTensor = torch.cat([targetUnitsTensor, targetUnits1, targetUnits2])

        meta = torch.zeros(META_SIZE)
        meta[0] = resources[0] / 100
        meta[1] = resources[1] / 100
        meta[2] = startingFood / 10
        meta[3] = targetFood / 10
        targetUnitTensors.append(targetUnitsTensor)
        metas.append(meta)
        offsets.append((offset, 50, 50))
        
    
    numInstances = len(targetUnitsList)
    startingUnitsTensor = startingUnitsTensor.expand([numInstances, -1])
    targetUnitsTensor = torch.stack(targetUnitTensors)
    metaTensors = torch.stack(metas)

    assert metaTensors.shape == (numInstances, META_SIZE)
    assert startingUnitsTensor.shape == (numInstances, STARTING_UNIT_TENSOR_SIZE)
    assert targetUnitsTensor.shape == (numInstances, TARGET_UNIT_TENSOR_SIZE)
    with torch.no_grad():
        net.eval()
        result = net(startingUnitsTensor, targetUnitsTensor, metaTensors).numpy()
        return np.maximum(0.0, result) / np.array([[score_scale, score_scale_minerals, score_scale_vespene]]) + np.array(offsets)

def load_session(s):
    data = json.loads(s)
    for item in data["instances"]:
        load_instance(item)

def load_all():
    print("Loading training data...")
    fs = os.listdir(data_path)
    # fs = natural_sort(fs)
    random.shuffle(fs)
    # fs = fs[1:1000]
    for p in fs:
        f = open(data_path + "/" + p)
        s = f.read()
        f.close()
        load_session(s)
    
    print("Loaded " + str(len(memory)) + " instances")
    time.sleep(1)
    print("Done")

class Net(nn.Module):
    # def add(self, layer):
    #     self.layers.append(layer)

    def __init__(self):
        N = 50
        M1 = N * 4
        M3 = N * 2
        M4 = N
        M2 = 10
        super(Net, self).__init__()
        self.fc1_1 = nn.Linear(STARTING_UNIT_TENSOR_SIZE, M1)
        self.fc1_2 = nn.Linear(TARGET_UNIT_TENSOR_SIZE, M1)
        layers = []
        layers.append(nn.Linear(M1 * 2 + META_SIZE, M3))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(M3))
        layers.append(nn.Linear(M3, M4))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(M4))

        layers.append(nn.Linear(M4, M2))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(M2))

        layers.append(nn.Linear(M2, M2))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(M2, M2))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(M2))

        layers.append(nn.Linear(M2, M2))
        layers.append(nn.LeakyReLU())
        
        layers.append(nn.BatchNorm1d(M2))

        layers.append(nn.Linear(M2, M2))
        layers.append(nn.LeakyReLU())
        layers.append(nn.BatchNorm1d(M2))

        # layers.append(nn.Linear(M2, M2))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.Linear(M2, M2))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.Linear(M2, M2))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.Linear(M2, M2))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.Linear(M2, M2))
        # layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(M2, 5))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(5, 5))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(5, 3))
        layers.append(nn.LeakyReLU())
        self.seq = nn.Sequential(*layers)

    def forward(self, a, b, meta):
        a1 = F.leaky_relu(self.fc1_1(a))
        b1 = F.leaky_relu(self.fc1_2(b))
        c = torch.cat([a1,b1,meta], dim=1)
        c = self.seq(c)
        # for layer in self.layers:
        #     c = layer(c)
        
        # c = F.leaky_relu(self.fc2(c))
        # c = self.norm1(c)
        # c = F.leaky_relu(self.fc3(c))
        # c = F.leaky_relu(self.fc4(c))
        # c = F.leaky_relu(self.fc5(c))
        # # c = self.drop(c)

        # c = F.leaky_relu(self.fc5_1(c))
        # c = self.norm2(c)
        # c = F.leaky_relu(self.fc5_2(c))
        # c = F.leaky_relu(self.fc5_3(c))
        # # c = F.leaky_relu(self.fc5_4(c))
        # # c = F.leaky_relu(self.fc5_5(c))
        # # c = F.leaky_relu(self.fc5_6(c))
        # # c = F.leaky_relu(self.fc5_7(c))
        # # c = F.leaky_relu(self.fc5_8(c))

        # c = F.leaky_relu(self.fc6(c))
        # c = F.leaky_relu(self.fc7(c))
        # c = F.leaky_relu(self.fc8(c))
        return c

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        M1 = NUM_UNITS * 2
        self.fc1_1 = nn.Linear(STARTING_UNIT_TENSOR_SIZE, M1)
        self.fc1_2 = nn.Linear(TARGET_UNIT_TENSOR_SIZE, M1)

        self.lin1 = nn.Linear(M1 * 2 + META_SIZE, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 32)
        self.lin4 = nn.Linear(32, 32)
        self.lin5 = nn.Linear(32, 3)
    
    def forward(self, a, b, meta):
        a1 = F.leaky_relu(self.fc1_1(a))
        b1 = F.leaky_relu(self.fc1_2(b))
        c = torch.cat([a1,b1,meta], dim=1)

        x = F.leaky_relu(self.lin1(c))
        y = x
        y = F.leaky_relu(self.lin2(y))
        y = F.leaky_relu(self.lin3(y))
        x = x + y
        y = x
        y = x + F.leaky_relu(self.lin4(y))
        y = F.leaky_relu(self.lin5(y))
        return y



net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)

trainLoader = None
testLoader = None

losses = []
weighted_losses = []
test_losses = []
test_losses_w = []
last_scatter_x = np.array([])
last_scatter_y = np.array([])
last_scatter_x2 = np.array([])
last_scatter_y2 = np.array([])
last_scatter_x3 = np.array([])
last_scatter_y3 = np.array([])

train_last_scatter_x = np.array([])
train_last_scatter_y = np.array([])
train_last_scatter_x2 = np.array([])
train_last_scatter_y2 = np.array([])
train_last_scatter_x3 = np.array([])
train_last_scatter_y3 = np.array([])

lossModel = nn.MSELoss()
score_scale = 1/1000
score_scale_minerals = 1/5000
score_scale_vespene = 1/2000
score_weight_shape = 50

def split_data():
    global trainLoader, testLoader
    test_split = 0.2
    test_length = int(round(len(memory) * test_split))
    [test_data, train_data] = torch.utils.data.random_split(memory, [test_length, len(memory) - test_length])
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)
    testLoader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

def optimize_model(doTrain=True):
    if trainLoader is None:
        split_data()

    global last_scatter_x, last_scatter_y, last_scatter_x2, last_scatter_y2, last_scatter_x3, last_scatter_y3, train_last_scatter_x, train_last_scatter_y, train_last_scatter_x2, train_last_scatter_y2, train_last_scatter_x3, train_last_scatter_y3

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss_w = 0.0
        loss_counter = 0
        net.train()

        train_last_scatter_x = np.array([])
        train_last_scatter_y = np.array([])
        train_last_scatter_x2 = np.array([])
        train_last_scatter_y2 = np.array([])
        train_last_scatter_x3 = np.array([])
        train_last_scatter_y3 = np.array([])
        
        for i, data in enumerate(trainLoader, 0):
            # get the inputs
            startingUnits, targetUnits, metas, targetTimes, origDataIndices, targetMineralCost, targetVespeneCost = data
            targetTimes = targetTimes.to(dtype=torch.float) * score_scale
            targetMineralCost = targetMineralCost.to(dtype=torch.float) * score_scale_minerals
            targetVespeneCost = targetVespeneCost.to(dtype=torch.float) * score_scale_vespene
            targets = torch.stack([targetTimes, targetMineralCost, targetVespeneCost], dim=1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(startingUnits, targetUnits, metas)
            # outputs = outputs.reshape(outputs.shape[0])
            item_losses = (outputs - targets)**2
            item_losses = item_losses.sum(dim=1)
            item_losses_w = item_losses * (score_weight_shape*score_scale) / (targetTimes + score_weight_shape*score_scale)
            loss = item_losses_w.mean()
            unweighted_loss = item_losses.mean()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += math.sqrt(unweighted_loss.item())
            running_loss_w += math.sqrt(loss.item())
            loss_counter += 1


            train_last_scatter_x = np.concatenate([train_last_scatter_x, targetTimes.numpy() / score_scale])
            train_last_scatter_y = np.concatenate([train_last_scatter_y, outputs[:,0].detach().numpy() / score_scale])

            train_last_scatter_x2 = np.concatenate([train_last_scatter_x2, targetMineralCost.numpy() / score_scale_minerals])
            train_last_scatter_y2 = np.concatenate([train_last_scatter_y2, outputs[:,1].detach().numpy() / score_scale_minerals])

            train_last_scatter_x3 = np.concatenate([train_last_scatter_x3, targetVespeneCost.numpy() / score_scale_vespene])
            train_last_scatter_y3 = np.concatenate([train_last_scatter_y3, outputs[:,2].detach().numpy() / score_scale_vespene])
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #     (epoch + 1, i + 1, running_loss / 2000))
                # running_loss = 0.0
                pass

        mean_loss = (running_loss / loss_counter) / score_scale
        mean_loss_w = (running_loss_w / loss_counter) / score_scale
        losses.append(mean_loss)
        weighted_losses.append(mean_loss_w)
        test_network()
        print(mean_loss)

def test_network():
    global last_scatter_x, last_scatter_y, last_scatter_x2, last_scatter_y2, last_scatter_x3, last_scatter_y3, train_last_scatter_x, train_last_scatter_y, train_last_scatter_x2, train_last_scatter_y2, train_last_scatter_x3, train_last_scatter_y3
    with torch.no_grad():
        net.eval()

        test_loss = 0
        test_loss_w = 0
        test_loss_counter = 0
        last_scatter_x = np.array([])
        last_scatter_y = np.array([])
        last_scatter_x2 = np.array([])
        last_scatter_y2 = np.array([])
        last_scatter_x3 = np.array([])
        last_scatter_y3 = np.array([])
        error_times = 0
        error_times_cnt = 0

        for data in testLoader:
            startingUnits, targetUnits, metas, targetTimes, origDataIndices, targetMineralCost, targetVespeneCost = data
            targetTimes = targetTimes.to(dtype=torch.float) * score_scale
            targetMineralCost = targetMineralCost.to(dtype=torch.float) * score_scale_minerals
            targetVespeneCost = targetVespeneCost.to(dtype=torch.float) * score_scale_vespene
            targets = torch.stack([targetTimes, targetMineralCost, targetVespeneCost], dim=1)

            # forward + backward + optimize
            outputs = net(startingUnits, targetUnits, metas)

            last_scatter_x = np.concatenate([last_scatter_x, targetTimes.numpy() / score_scale])
            last_scatter_y = np.concatenate([last_scatter_y, outputs[:,0].numpy() / score_scale])

            last_scatter_x2 = np.concatenate([last_scatter_x2, targetMineralCost.numpy() / score_scale_minerals])
            last_scatter_y2 = np.concatenate([last_scatter_y2, outputs[:,1].numpy() / score_scale_minerals])

            last_scatter_x3 = np.concatenate([last_scatter_x3, targetVespeneCost.numpy() / score_scale_vespene])
            last_scatter_y3 = np.concatenate([last_scatter_y3, outputs[:,2].numpy() / score_scale_vespene])

            item_losses = (outputs - targets)**2
            error_times += item_losses[:, 0].sum().item()
            error_times_cnt += item_losses.shape[0]
            item_losses = item_losses.sum(dim=1)
            item_losses_w = item_losses * (score_weight_shape*score_scale) / (targetTimes + score_weight_shape*score_scale)
            loss = item_losses_w.mean()
            unweighted_loss = item_losses.mean()
            
            test_loss_w += math.sqrt(loss.item())
            test_loss += math.sqrt(unweighted_loss.item())
            test_loss_counter += 1

            worstIndex = item_losses_w.argmax()
            print("Worst index " + str(worstIndex))
            print("Expected time " + str(outputs[worstIndex]/score_scale))
            print(originalDatas[origDataIndices[worstIndex].item()])

        
        test_loss /= test_loss_counter * score_scale
        test_loss_w /= test_loss_counter * score_scale
        error_times = math.sqrt(error_times/error_times_cnt) / score_scale
        test_losses.append(test_loss)
        test_losses_w.append(test_loss_w)
        print(test_loss, test_loss_w, error_times)

        # startingUnits, targetUnits, meta, targetTime = random.choice(memory.data)
        # outputs = net(startingUnits.reshape(1,-1), targetUnits.reshape(1,-1), meta.reshape(1,-1))
        # print(f"Actual: {targetTime}, Predicted: {outputs.item()/score_scale}")


episode = 0

epss = []
temps = []

def plot_loss():
    durations_t = torch.tensor(losses, dtype=torch.float)
    fig = plt.figure(1, [20, 10])
    plt.clf()
    ax = fig.add_subplot(2,4,1)
    plt.title('Training...')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.set_yscale('log')
    plt.plot(durations_t.numpy(), label="Train Loss")
    plt.plot(torch.tensor(test_losses, dtype=torch.float).numpy(), label="Test Loss")
    plt.plot(torch.tensor(test_losses_w, dtype=torch.float).numpy(), label="Weighted Test Loss")
    plt.plot(torch.tensor(weighted_losses, dtype=torch.float).numpy(), label="Weighted Train Loss")
    plt.legend()
    plt.ylim([10**1, 10**3])
    plt.grid(True, which='both', axis='y')


    # samples = np.array([last_scatter_x.transpose(), last_scatter_y.transpose()])
    # densObj = kde(samples)
    # def makeColours( vals ):
    #     colours = np.zeros( (len(vals),3) )
    #     norm = matplotlib.colors.Normalize( vmin=vals.min(), vmax=vals.max() )

    #     #Can put any colormap you like here.
    #     colours = [cm.ScalarMappable( norm=norm, cmap='inferno').to_rgba( val ) for val in vals]

    #     return colours

    # sampleColors = makeColours( densObj.evaluate( samples ) )

    coords = [
        (2, 4, 2), (2, 4, 3), (2, 4, 4),
        (2, 4, 6), (2, 4, 7), (2, 4, 8)
    ]
    datas = [
        (last_scatter_x, last_scatter_y), (last_scatter_x2, last_scatter_y2), (last_scatter_x3, last_scatter_y3),
        (train_last_scatter_x, train_last_scatter_y), (train_last_scatter_x2, train_last_scatter_y2), (train_last_scatter_x3, train_last_scatter_y3)
    ]
    labels = [
        'Estimated Build Order Times [s]', 'Estimated Mineral Costs', 'Estimated Vepene Costs [s]',
        'Estimated Build Order Times [s] train', 'Estimated Mineral Costs train', 'Estimated Vepene Costs [s] train',
    ]
    for i in range(6):
        ax = fig.add_subplot(*coords[i])
        data = datas[i]
        plt.scatter(x=data[0], y=data[1], s=1, c="#e41a1c", alpha=0.2)
        plt.axis('scaled')
        mx = max(data[0].max(), data[1].max())
        plt.plot([0, mx], [0, mx], c="#000000", alpha=0.5)
        plt.ylim(bottom=-1, top=mx)
        plt.xlim(left=-1, right=mx)
        plt.ylabel(labels[i])
        plt.xlabel('Ground Truth Build Order Times [s]')

    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_paper_loss():
    durations_t = torch.tensor(losses, dtype=torch.float)
    fig = plt.figure(1, [10, 3])
    plt.clf()

    # samples = np.array([last_scatter_x.transpose(), last_scatter_y.transpose()])
    # densObj = kde(samples)
    # def makeColours( vals ):
    #     colours = np.zeros( (len(vals),3) )
    #     norm = matplotlib.colors.Normalize( vmin=vals.min(), vmax=vals.max() )

    #     #Can put any colormap you like here.
    #     colours = [cm.ScalarMappable( norm=norm, cmap='inferno').to_rgba( val ) for val in vals]

    #     return colours

    # sampleColors = makeColours( densObj.evaluate( samples ) )

    coords = [
        (1, 3, 1), (1, 3, 2), (1, 3, 3),
    ]
    datas = [
        (last_scatter_x, last_scatter_y), (last_scatter_x2, last_scatter_y2), (last_scatter_x3, last_scatter_y3),
    ]
    ylabels = [
        'Estimated Build Order Time [s]', 'Estimated Mineral Cost', 'Estimated Vespene Cost',
    ]
    xlabels = [
        'Ground Truth Build Order Time [s]', 'Ground Truth Mineral Cost', 'Ground Truth Vespene Cost',
    ]
    for i in range(3):
        ax = fig.add_subplot(*coords[i])
        data = datas[i]
        plt.scatter(x=data[0], y=data[1], s=1, c="#e41a1c", alpha=0.1, rasterized=True)
        plt.axis('scaled')
        mx = max(data[0].max(), data[1].max())
        plt.plot([0, mx], [0, mx], c="#000000", alpha=0.5)
        plt.ylim(bottom=-1, top=mx)
        plt.xlim(left=-1, right=mx)
        plt.ylabel(ylabels[i])
        plt.xlabel(xlabels[i])

    plt.subplots_adjust(left=0.1, bottom=0.17, right=0.94, top=0.90, wspace=0.38, hspace=0.24)    
    # plt.pause(40)  # pause a bit so that plots are updated
    # pdf = PdfPages(f"/Users/arong/cloud/Skolarbeten/ML-2/thesis/draft/graphics/generated/bo_nn_optimizer_train.pdf")
    # pdf.savefig(fig, dpi=200)
    # pdf.close()
    plt.savefig(f"/Users/arong/cloud/Skolarbeten/ML-2/thesis/draft/graphics/generated/bo_nn_optimizer_train.pdf", dpi=100)

    # 

def optimize(steps: int):
    global episode
    for i in range(steps):
        optimize_model()
        episode += 1
        # if episode % TARGET_UPDATE == 0:
        #     target_net.load_state_dict(policy_net.state_dict())

    plot_loss()
    # plt.show()



# num_episodes = 50
# for i_episode in range(num_episodes):
#     # Initialize the environment and state
#     env.reset()
#     last_screen = get_screen()
#     current_screen = get_screen()
#     state = current_screen - last_screen
#     for t in count():
#         # Select and perform an action
#         action = select_action(state)
#         _, reward, done, _ = env.step(action.item())
#         reward = torch.tensor([reward], device=device)

#         # Observe new state
#         last_screen = current_screen
#         current_screen = get_screen()
#         if not done:
#             next_state = current_screen - last_screen
#         else:
#             next_state = None

#         # Store the transition in memory
#         memory.push(state, action, next_state, reward)

#         # Move to the next state
#         state = next_state

#         # Perform one step of the optimization (on the target network)
#         optimize_model()
#         if done:
#             episode_durations.append(t + 1)
#             plot_durations()
#             break
#     # Update the target network
#     if i_episode % TARGET_UPDATE == 0:
#         target_net.load_state_dict(policy_net.state_dict())

# print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.show()


if __name__ == "__main__":
    # Only import matplotlib when in training mode
    # (can mess things up if matplotlib is imported multiple times from C++)
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import rcParams
    plt.ioff()
    import argparse

    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['CMU Serif']
    rcParams['font.size'] = 14

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.train:
        load_all()
        for i in range(150):
            optimize(1)
    
        torch.save(net.state_dict(), "models/buildorders_mean.weights")
    
    if args.visualize:
        net.load_state_dict(torch.load("models/buildorders.weights"))
        print(predict([
            (unitLookup.findByName("PROTOSS_NEXUS").type_ids[0], 1),
            (unitLookup.findByName("PROTOSS_PROBE").type_ids[0], 16),
            (unitLookup.findByName("PROTOSS_PYLON").type_ids[0], 2),
            (unitLookup.findByName("PROTOSS_GATEWAY").type_ids[0], 2),
        ], [50, 0], [
            [
                (unitLookup.findByName("PROTOSS_ZEALOT").type_ids[0], 6),
                (unitLookup.findByName("PROTOSS_IMMORTAL").type_ids[0], 4),
            ],
            [
                (unitLookup.findByName("PROTOSS_ZEALOT").type_ids[0], 6),
                (unitLookup.findByName("PROTOSS_IMMORTAL").type_ids[0], 1),
            ],
            [
                (unitLookup.findByName("PROTOSS_ZEALOT").type_ids[0], 1),
                (unitLookup.findByName("PROTOSS_IMMORTAL").type_ids[0], 1),
            ],
            [
                (unitLookup.findByName("PROTOSS_COLOSSUS").type_ids[0], 2),
                (unitLookup.findByName("PROTOSS_OBSERVER").type_ids[0], 0),
            ],
            [
                (unitLookup.findByName("PROTOSS_COLOSSUS").type_ids[0], 2),
                (unitLookup.findByName("PROTOSS_OBSERVER").type_ids[0], 1),
            ],
            [
                (unitLookup.findByName("PROTOSS_COLOSSUS").type_ids[0], 2),
                (unitLookup.findByName("PROTOSS_OBSERVER").type_ids[0], 0),
            ],
            [
                (unitLookup.findByName("PROTOSS_COLOSSUS").type_ids[0], 2),
                (unitLookup.findByName("PROTOSS_OBSERVER").type_ids[0], 2),
            ],
            [
                (unitLookup.findByName("PROTOSS_COLOSSUS").type_ids[0], 2),
                (unitLookup.findByName("PROTOSS_OBSERVER").type_ids[0], 3),
            ],
            [
                (unitLookup.findByName("PROTOSS_COLOSSUS").type_ids[0], 2),
                (unitLookup.findByName("PROTOSS_OBSERVER").type_ids[0], 4),
            ],
            [
                (unitLookup.findByName("PROTOSS_COLOSSUS").type_ids[0], 2),
                (unitLookup.findByName("PROTOSS_OBSERVER").type_ids[0], 5),
            ],
            [
                (unitLookup.findByName("PROTOSS_COLOSSUS").type_ids[0], 2),
                (unitLookup.findByName("PROTOSS_OBSERVER").type_ids[0], 6),
            ],
            [
                (unitLookup.findByName("WARPGATERESEARCH").type_ids[0], 1),
            ],
        ]))

        print(unitLookup.findByName("PROTOSS_GATEWAY").mineral_cost + unitLookup.findByName("PROTOSS_CYBERNETICSCORE").mineral_cost + unitLookup.findByName("PROTOSS_STARGATE").mineral_cost + unitLookup.findByName("PROTOSS_FLEETBEACON").mineral_cost)
    
    if args.plot:
        plt.ion()
        load_all()
        net.load_state_dict(torch.load("models/buildorders_mean.weights"))
        split_data()
        test_network()
        plot_paper_loss()
else:
    print("Loading weights")
    net.load_state_dict(torch.load("models/buildorders_mean.weights"))
