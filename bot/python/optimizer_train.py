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
data_path = "training_data/buildorders_time/1"

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


unitIndexMapTerran = {
    29: 0,  # TERRAN_ARMORY
    55: 1,  # TERRAN_BANSHEE
    21: 2,  # TERRAN_BARRACKS
    38: 3,  # TERRAN_BARRACKSREACTOR
    37: 4,  # TERRAN_BARRACKSTECHLAB
    57: 5,  # TERRAN_BATTLECRUISER
    24: 6,  # TERRAN_BUNKER
    18: 7,  # TERRAN_COMMANDCENTER
    692: 8,  # TERRAN_CYCLONE
    22: 9,  # TERRAN_ENGINEERINGBAY
    27: 10,  # TERRAN_FACTORY
    40: 11,  # TERRAN_FACTORYREACTOR
    39: 12,  # TERRAN_FACTORYTECHLAB
    30: 13,  # TERRAN_FUSIONCORE
    50: 14,  # TERRAN_GHOST
    26: 15,  # TERRAN_GHOSTACADEMY
    53: 16,  # TERRAN_HELLION
    484: 17,  # TERRAN_HELLIONTANK
    689: 18,  # TERRAN_LIBERATOR
    51: 19,  # TERRAN_MARAUDER
    48: 20,  # TERRAN_MARINE
    54: 21,  # TERRAN_MEDIVAC
    23: 22,  # TERRAN_MISSILETURRET
    268: 23,  # TERRAN_MULE
    132: 24,  # TERRAN_ORBITALCOMMAND
    130: 25,  # TERRAN_PLANETARYFORTRESS
    56: 26,  # TERRAN_RAVEN
    49: 27,  # TERRAN_REAPER
    20: 28,  # TERRAN_REFINERY
    45: 29,  # TERRAN_SCV
    25: 30,  # TERRAN_SENSORTOWER
    32: 31,   # TERRAN_SIEGETANKSIEGED
    33: 31,  # TERRAN_SIEGETANK
    28: 32,  # TERRAN_STARPORT
    42: 33,  # TERRAN_STARPORTREACTOR
    41: 34,  # TERRAN_STARPORTTECHLAB
    19: 35,  # TERRAN_SUPPLYDEPOT
    52: 36,  # TERRAN_THOR
    691: 36, # TERRAN_THORAP
    34: 37,   # TERRAN_VIKINGASSAULT
    35: 37,  # TERRAN_VIKINGFIGHTER
    498: 38,  # TERRAN_WIDOWMINE,
}

economicallyRelevantUnitsTerran = [
    29,  # TERRAN_ARMORY
    21,  # TERRAN_BARRACKS
    38,  # TERRAN_BARRACKSREACTOR
    37,  # TERRAN_BARRACKSTECHLAB
    24,  # TERRAN_BUNKER
    18,  # TERRAN_COMMANDCENTER
    22,  # TERRAN_ENGINEERINGBAY
    27,  # TERRAN_FACTORY
    40,  # TERRAN_FACTORYREACTOR
    39,  # TERRAN_FACTORYTECHLAB
    30,  # TERRAN_FUSIONCORE
    26,  # TERRAN_GHOSTACADEMY
    23,  # TERRAN_MISSILETURRET
    132,  # TERRAN_ORBITALCOMMAND
    130,  # TERRAN_PLANETARYFORTRESS
    20,  # TERRAN_REFINERY
    45,  # TERRAN_SCV
    25,  # TERRAN_SENSORTOWER
    28,  # TERRAN_STARPORT
    42,  # TERRAN_STARPORTREACTOR
    41,  # TERRAN_STARPORTTECHLAB
    19,  # TERRAN_SUPPLYDEPOT
]

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

unitLookup = mappings.UnitLookup([u for u in mappings.protossUnits if u[0] in units])
economicUnitLookup = mappings.UnitLookup([u for u in mappings.protossUnits if u[0] in economicallyRelevantUnitsProtoss])

# economicallyRelevantUnitsMap = {}
# for i in range(len(economicallyRelevantUnits)):
    # economicallyRelevantUnitsMap[economicallyRelevantUnits[i]] = i

# economicallyRelevantUnitsMap = unit_index_map

# unitFoodRequirement = {
#     29: 0,  # TERRAN_ARMORY
#     55: -3,  # TERRAN_BANSHEE
#     21: 0,  # TERRAN_BARRACKS
#     57: -6,  # TERRAN_BATTLECRUISER
#     24: 0,  # TERRAN_BUNKER
#     18: 15,  # TERRAN_COMMANDCENTER
#     692: -3,  # TERRAN_CYCLONE
#     22: 0,  # TERRAN_ENGINEERINGBAY
#     27: 0,  # TERRAN_FACTORY
#     30: 0,  # TERRAN_FUSIONCORE
#     50: -2,  # TERRAN_GHOST
#     26: 0,  # TERRAN_GHOSTACADEMY
#     53: -2,  # TERRAN_HELLION
#     484: -2,  # TERRAN_HELLIONTANK
#     689: -3,  # TERRAN_LIBERATOR
#     51: -2,  # TERRAN_MARAUDER
#     48: -1,  # TERRAN_MARINE
#     54: -2,  # TERRAN_MEDIVAC
#     23: 0,  # TERRAN_MISSILETURRET
#     132: 15,  # TERRAN_ORBITALCOMMAND
#     130: 15,  # TERRAN_PLANETARYFORTRESS
#     56: -2,  # TERRAN_RAVEN
#     49: -1,  # TERRAN_REAPER
#     20: 0,  # TERRAN_REFINERY
#     45: -1,  # TERRAN_SCV
#     25: 0,  # TERRAN_SENSORTOWER
#     32: -3,   # TERRAN_SIEGETANKSIEGED
#     33: -3,  # TERRAN_SIEGETANK
#     28: 0,  # TERRAN_STARPORT
#     19: 8,  # TERRAN_SUPPLYDEPOT
#     52: -6,  # TERRAN_THOR
#     34: -2,   # TERRAN_VIKINGASSAULT
#     35: -2,  # TERRAN_VIKINGFIGHTER
#     498: -2,  # TERRAN_WIDOWMINE
# }

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

    if allStartingUnits.sum() > 200:
        # print("Skipping item with", startingUnits.sum(), "units")
        return
    
    # print(targetUnits)
    # print(startingFood)
    targetUnits = torch.max(torch.tensor(0.0), targetUnits - allStartingUnits)
    targetFood = (targetUnits * unitFoodRequirementTensor).sum()
    startingFood = (allStartingUnits * unitFoodRequirementTensor).sum()

    startingUnits1 = (startingUnits > 0).to(dtype=torch.float)
    targetUnits1 = (targetUnits > 0).to(dtype=torch.float)
    startingUnits2 = (startingUnits == 1).to(dtype=torch.float)
    targetUnits2 = (targetUnits == 1).to(dtype=torch.float)

    startingUnits = torch.cat([startingUnits, startingUnits1, startingUnits2])
    targetUnits = torch.cat([targetUnits, targetUnits1, targetUnits2])

    meta = torch.zeros(META_SIZE)
    meta[0] = item["startingMinerals"]
    meta[1] = item["startingVespene"]
    meta[2] = startingFood
    meta[3] = targetFood

    # meta[2] = item["buildOrderTime"]

    assert meta.shape == (META_SIZE,)
    assert startingUnits.shape == (STARTING_UNIT_TENSOR_SIZE,)
    assert targetUnits.shape == (TARGET_UNIT_TENSOR_SIZE,)
    originalDatas.append(item)
    memory.add((startingUnits, targetUnits, meta, item["buildOrderTime"], len(originalDatas) - 1))

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
    for targetUnits in targetUnitsList:
        targetUnitsTensor = torch.zeros(NUM_UNITS, dtype=torch.float)
        for u in targetUnits:
            targetUnitsTensor[unitLookup.unit_index_map[u[0]]] += u[1]

        targetFood = (targetUnitsTensor * unitFoodRequirementTensor).sum()
        targetUnits1 = (targetUnitsTensor > 0).to(dtype=torch.float)
        targetUnits2 = (targetUnitsTensor == 1).to(dtype=torch.float)
        targetUnitsTensor = torch.cat([targetUnitsTensor, targetUnits1, targetUnits2])

        meta = torch.zeros(META_SIZE)
        meta[0] = resources[0]
        meta[1] = resources[1]
        meta[2] = startingFood
        meta[3] = targetFood
        targetUnitTensors.append(targetUnitsTensor)
        metas.append(meta)
        
    
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
        return np.maximum(0.0, result) / score_scale

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
        M1 = NUM_UNITS * 2
        M3 = NUM_UNITS * 1
        M4 = NUM_UNITS
        M2 = 20
        super(Net, self).__init__()
        self.fc1_1 = nn.Linear(STARTING_UNIT_TENSOR_SIZE, M1)
        self.fc1_2 = nn.Linear(TARGET_UNIT_TENSOR_SIZE, M1)
        layers = []
        layers.append(nn.Linear(M1 * 2 + META_SIZE, M3))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(M3))
        layers.append(nn.Linear(M3, M4))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(M4))

        layers.append(nn.Linear(M4, M2))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(M2))

        layers.append(nn.Linear(M2, M2))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(M2, M2))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(M2))

        layers.append(nn.Linear(M2, M2))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(M2))

        layers.append(nn.Linear(M2, M2))
        layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(M2))

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
        layers.append(nn.Linear(5, 2))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(2, 1))
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
lossModel = nn.MSELoss()
score_scale = 1/1000
score_weight_shape = 50

def split_data():
    global trainLoader, testLoader
    test_split = 0.2
    test_length = int(round(len(memory) * test_split))
    [test_data, train_data] = torch.utils.data.random_split(memory, [test_length, len(memory) - test_length])
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True, num_workers=0)
    testLoader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

def optimize_model():
    if trainLoader is None:
        split_data()

    global last_scatter_x, last_scatter_y

    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss_w = 0.0
        loss_counter = 0
        net.train()
        for i, data in enumerate(trainLoader, 0):
            # get the inputs
            startingUnits, targetUnits, metas, targetTimes, origDataIndices = data
            targetTimes = targetTimes.to(dtype=torch.float) * score_scale

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(startingUnits, targetUnits, metas)
            outputs = outputs.reshape(outputs.shape[0])
            item_losses = (outputs - targetTimes)**2
            item_losses_w = item_losses * (score_weight_shape*score_scale) / (targetTimes + score_weight_shape*score_scale)
            loss = item_losses_w.mean()
            unweighted_loss = item_losses.mean()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += math.sqrt(unweighted_loss.item())
            running_loss_w += math.sqrt(loss.item())
            loss_counter += 1
            if i % 2000 == 1999:    # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #     (epoch + 1, i + 1, running_loss / 2000))
                # running_loss = 0.0
                pass

        mean_loss = (running_loss / loss_counter) / score_scale
        mean_loss_w = (running_loss_w / loss_counter) / score_scale

        with torch.no_grad():
            net.eval()

            test_loss = 0
            test_loss_w = 0
            test_loss_counter = 0
            last_scatter_x = np.array([])
            last_scatter_y = np.array([])
            for data in testLoader:
                startingUnits, targetUnits, metas, targetTimes, origDataIndices = data
                targetTimes = targetTimes.to(dtype=torch.float) * score_scale

                # forward + backward + optimize
                outputs = net(startingUnits, targetUnits, metas)
                outputs = outputs.reshape(outputs.shape[0])

                last_scatter_x = np.concatenate([last_scatter_x, targetTimes.numpy() / score_scale])
                last_scatter_y = np.concatenate([last_scatter_y, outputs.numpy() / score_scale])

                item_losses = (outputs - targetTimes)**2
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
            print(mean_loss, test_loss)
            losses.append(mean_loss)
            weighted_losses.append(mean_loss_w)
            test_losses.append(test_loss)
            test_losses_w.append(test_loss_w)

            # startingUnits, targetUnits, meta, targetTime = random.choice(memory.data)
            # outputs = net(startingUnits.reshape(1,-1), targetUnits.reshape(1,-1), meta.reshape(1,-1))
            # print(f"Actual: {targetTime}, Predicted: {outputs.item()/score_scale}")


episode = 0

epss = []
temps = []

def plot_loss():
    durations_t = torch.tensor(losses, dtype=torch.float)
    fig = plt.figure(1, [5, 10])
    plt.clf()
    ax = fig.add_subplot(2,1,1)
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

    ax = fig.add_subplot(2,1,2)
    plt.scatter(x=last_scatter_x, y=last_scatter_y, s=1, c="#e41a1c", alpha=0.2)
    # plt.scatter(x=last_scatter_x, y=last_scatter_y, s=1, color=sampleColors, alpha=0.4)
    plt.axis('scaled')
    mx = max(last_scatter_x.max(), last_scatter_y.max())
    plt.plot([0, mx], [0, mx], c="#000000", alpha=0.5)
    plt.ylim(bottom=-1, top=mx)
    plt.xlim(left=-1, right=mx)
    plt.ylabel('Estimated Build Order Times [s]')
    plt.xlabel('Ground Truth Build Order Times [s]')

    plt.pause(0.001)  # pause a bit so that plots are updated



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
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.ioff()

    load_all()
    for i in range(50):
        optimize(1)
    
    torch.save(net.state_dict(), "models/buildorders.weights")
else:
    print("Loading weights")
    net.load_state_dict(torch.load("models/buildorders.weights"))
