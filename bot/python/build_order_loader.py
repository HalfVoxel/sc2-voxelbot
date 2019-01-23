import math
import random
import numpy as np
import json
from replay_memory import Transition
from mappings import UnitLookup, ignoredUnits


# unitNameMap = {
#     29: "TERRAN_ARMORY",
#     55: "TERRAN_BANSHEE",
#     21: "TERRAN_BARRACKS",
#     38: "TERRAN_BARRACKSREACTOR",
#     37: "TERRAN_BARRACKSTECHLAB",
#     57: "TERRAN_BATTLECRUISER",
#     24: "TERRAN_BUNKER",
#     18: "TERRAN_COMMANDCENTER",
#     692: "TERRAN_CYCLONE",
#     22: "TERRAN_ENGINEERINGBAY",
#     27: "TERRAN_FACTORY",
#     40: "TERRAN_FACTORYREACTOR",
#     39: "TERRAN_FACTORYTECHLAB",
#     30: "TERRAN_FUSIONCORE",
#     50: "TERRAN_GHOST",
#     26: "TERRAN_GHOSTACADEMY",
#     53: "TERRAN_HELLION",
#     484: "TERRAN_HELLIONTANK",
#     689: "TERRAN_LIBERATOR",
#     51: "TERRAN_MARAUDER",
#     48: "TERRAN_MARINE",
#     54: "TERRAN_MEDIVAC",
#     23: "TERRAN_MISSILETURRET",
#     132: "TERRAN_ORBITALCOMMAND",
#     130: "TERRAN_PLANETARYFORTRESS",
#     56: "TERRAN_RAVEN",
#     49: "TERRAN_REAPER",
#     20: "TERRAN_REFINERY",
#     45: "TERRAN_SCV",
#     25: "TERRAN_SENSORTOWER",
#     32: "TERRAN_SIEGETANKSIEGED",
#     33: "TERRAN_SIEGETANK",
#     28: "TERRAN_STARPORT",
#     42: "TERRAN_STARPORTREACTOR",
#     41: "TERRAN_STARPORTTECHLAB",
#     19: "TERRAN_SUPPLYDEPOT",
#     52: "TERRAN_THOR",
#     691: "TERRAN_THORAP",
#     34: "TERRAN_VIKINGASSAULT",
#     35: "TERRAN_VIKINGFIGHTER",
#     498: "TERRAN_WIDOWMINE",
#     # 268: 23,  # TERRAN_MULE
# }

# self.unit_lookup.unit_index_map = {
#     29: 0,  # TERRAN_ARMORY
#     55: 1,  # TERRAN_BANSHEE
#     21: 2,  # TERRAN_BARRACKS
#     38: 3,  # TERRAN_BARRACKSREACTOR
#     37: 4,  # TERRAN_BARRACKSTECHLAB
#     57: 5,  # TERRAN_BATTLECRUISER
#     24: 6,  # TERRAN_BUNKER
#     18: 7,  # TERRAN_COMMANDCENTER
#     692: 8,  # TERRAN_CYCLONE
#     22: 9,  # TERRAN_ENGINEERINGBAY
#     27: 10,  # TERRAN_FACTORY
#     40: 11,  # TERRAN_FACTORYREACTOR
#     39: 12,  # TERRAN_FACTORYTECHLAB
#     30: 13,  # TERRAN_FUSIONCORE
#     50: 14,  # TERRAN_GHOST
#     26: 15,  # TERRAN_GHOSTACADEMY
#     53: 16,  # TERRAN_HELLION
#     484: 17,  # TERRAN_HELLIONTANK
#     689: 18,  # TERRAN_LIBERATOR
#     51: 19,  # TERRAN_MARAUDER
#     48: 20,  # TERRAN_MARINE
#     54: 21,  # TERRAN_MEDIVAC
#     23: 22,  # TERRAN_MISSILETURRET
#     132: 23,  # TERRAN_ORBITALCOMMAND
#     130: 24,  # TERRAN_PLANETARYFORTRESS
#     56: 25,  # TERRAN_RAVEN
#     49: 26,  # TERRAN_REAPER
#     20: 27,  # TERRAN_REFINERY
#     45: 28,  # TERRAN_SCV
#     25: 29,  # TERRAN_SENSORTOWER
#     32: 30,   # TERRAN_SIEGETANKSIEGED
#     33: 30,  # TERRAN_SIEGETANK
#     28: 31,  # TERRAN_STARPORT
#     42: 32,  # TERRAN_STARPORTREACTOR
#     41: 33,  # TERRAN_STARPORTTECHLAB
#     19: 34,  # TERRAN_SUPPLYDEPOT
#     52: 35,  # TERRAN_THOR
#     691: 35,  # TERRAN_THORAP
#     34: 36,   # TERRAN_VIKINGASSAULT
#     35: 36,  # TERRAN_VIKINGFIGHTER
#     498: 37,  # TERRAN_WIDOWMINE,
#     # 268: 23,  # TERRAN_MULE
# }

# unitIndexMap = {
#     21: 0,  # TERRAN_BARRACKS
#     38: 1,  # TERRAN_BARRACKSREACTOR
#     37: 2,  # TERRAN_BARRACKSTECHLAB
#     18: 3,  # TERRAN_COMMANDCENTER
#     48: 4,  # TERRAN_MARINE
#     45: 5,  # TERRAN_SCV
#     19: 6,  # TERRAN_SUPPLYDEPOT
#     20: 7,  # TERRAN_REFINERY
# }

# isUnitMilitary = {
#     29: False,  # TERRAN_ARMORY
#     55: True,  # TERRAN_BANSHEE
#     21: False,  # TERRAN_BARRACKS
#     57: True,  # TERRAN_BATTLECRUISER
#     24: False,  # TERRAN_BUNKER
#     18: False,  # TERRAN_COMMANDCENTER
#     692: True,  # TERRAN_CYCLONE
#     22: False,  # TERRAN_ENGINEERINGBAY
#     27: False,  # TERRAN_FACTORY
#     30: False,  # TERRAN_FUSIONCORE
#     50: True,  # TERRAN_GHOST
#     26: False,  # TERRAN_GHOSTACADEMY
#     53: True,  # TERRAN_HELLION
#     484: True,  # TERRAN_HELLIONTANK
#     689: True,  # TERRAN_LIBERATOR
#     51: True,  # TERRAN_MARAUDER
#     48: True,  # TERRAN_MARINE
#     54: True,  # TERRAN_MEDIVAC
#     23: False,  # TERRAN_MISSILETURRET
#     132: False,  # TERRAN_ORBITALCOMMAND
#     130: False,  # TERRAN_PLANETARYFORTRESS
#     56: True,  # TERRAN_RAVEN
#     49: True,  # TERRAN_REAPER
#     20: False,  # TERRAN_REFINERY
#     45: False,  # TERRAN_SCV
#     25: False,  # TERRAN_SENSORTOWER
#     32: True,   # TERRAN_SIEGETANKSIEGED
#     33: True,  # TERRAN_SIEGETANK
#     28: False,  # TERRAN_STARPORT
#     19: False,  # TERRAN_SUPPLYDEPOT
#     52: True,  # TERRAN_THOR
#     34: True,   # TERRAN_VIKINGASSAULT
#     35: True,  # TERRAN_VIKINGFIGHTER
#     498: True,  # TERRAN_WIDOWMINE
# }

# NUM_UNITS = len(set(unitIndexMap.values()))
# MILITARY_UNITS_MASK = np.zeros(NUM_UNITS)
# reverseUnitIndexMap = {}
# for k, v in unitIndexMap.items():
#     reverseUnitIndexMap[v] = k
#     MILITARY_UNITS_MASK[v] = 1 if k in isUnitMilitary and isUnitMilitary[k] else 0

# MILITARY_UNITS_MASK_INDICES = np.where(MILITARY_UNITS_MASK)[0]
# NON_MILITARY_UNITS_MASK_INDICES = np.where(MILITARY_UNITS_MASK==False)[0]

# print(f"Input tensor size: {TENSOR_INPUT_SIZE}")

# nonMilitaryIndexMap = {}
# for i in range(len(NON_MILITARY_UNITS_MASK_INDICES)):
#     for k, v in unitIndexMap.items():
#         if v == NON_MILITARY_UNITS_MASK_INDICES[i]:
#             nonMilitaryIndexMap[k] = i


class Statistics:
    def __init__(self):
        self.durations = []
        self.health_diffs = []
        self.total_rewards = []
        self.all_actions = []


def to_one_hot(tensor: np.array, dim: int):
    assert len(tensor.shape) == 1
    result = np.zeros(tensor.shape[0] * dim, dtype=np.float32)
    tensor = np.maximum(0, np.minimum(tensor, dim - 1))
    for i in range(tensor.shape[0]):
        result[i*dim + int(tensor[i])] = 1

    return result


class BuildOrderLoader:
    def __init__(self, unit_lookup: UnitLookup, gamma_per_second: float):
        self.gamma_per_second = gamma_per_second
        self.goalPool = []
        self.unit_lookup = unit_lookup
        self.num_units = self.unit_lookup.num_units
        self.tensor_input_size = self.get_tensor_size(self.unit_lookup.non_military_units_mask_indices, True)

    def get_tensor_size(self, unitIndices, include_goal):
        result = len(unitIndices)*3 + len(unitIndices)*10 + len(unitIndices)*5 + 7 + 8
        if include_goal:
            result += 2 * self.unit_lookup.num_units
        return result

    def calculate_session_rewards(self, session_json):
        session = json.loads(session_json)
        goal_tensor = self.createGoalTensor(session["goal"])
        states = session["states"]
        tensor_states = [self.createState(s) for s in states]
        rewards = []

        for i in range(len(states) - 1):
            deltaTime = states[i + 1]["time"] - states[i]["time"]
            reward = self.calculate_reward(tensor_states[i], tensor_states[i + 1], goal_tensor, deltaTime)
            rewards.append(reward)

        return rewards

    def createState(self, state, unitIndices=None):
        if unitIndices is None:
            unitIndices = self.unit_lookup.non_military_units_mask_indices

        unitCounts = np.zeros(self.num_units, dtype=np.float32)
        unitsAvailable = np.zeros(self.num_units, dtype=np.float32)
        unitsInProgress = np.zeros(self.num_units, dtype=np.float32)

        for unit in state["units"]:
            if unit["type"] in ignoredUnits:
                continue

            # Finished
            # TODO: Addon
            unitIndex = self.unit_lookup.unit_index_map[unit["type"]]
            unitCounts[unitIndex] += unit["totalCount"]
            unitsAvailable[unitIndex] += unit["availableCount"]

        for unit in state["unitsInProgress"]:
            if unit["type"] in ignoredUnits:
                continue

            # In progress
            unitIndex = self.unit_lookup.unit_index_map[unit["type"]]
            unitCounts[unitIndex] += 1
            unitsInProgress[unitIndex] += 1

        originalUnitCounts = unitCounts

        oneHotUnitsAvailable = to_one_hot(unitsAvailable[unitIndices], 3)
        oneHotUnitCounts = to_one_hot(unitCounts[unitIndices], 10)
        oneHotUnitsInProgress = to_one_hot(unitsInProgress[unitIndices], 5)

        # Some metadata, the data is normalized to approximately 1
        metaTensor = np.zeros(7, dtype=np.float32)
        metaTensor[0] = state["minerals"] / 100
        metaTensor[1] = state["vespene"] / 100
        metaTensor[2] = state["mineralsPerSecond"] / 10
        metaTensor[3] = state["vespenePerSecond"] / 10
        metaTensor[4] = state["highYieldMineralSlots"] / 10
        metaTensor[5] = state["lowYieldMineralSlots"] / 10
        harvesters = [45, 104, 116, 84]
        for h_type in harvesters:
            metaTensor[6] += unitCounts[self.unit_lookup.unit_index_map[h_type]] / 10  # SCV+Drone+Probe count
        foodTensor = to_one_hot(np.array([state["foodAvailable"]]), 8)

        # stateTensor = np.concatenate([np.array([state["time"]]), oneHotUnitCounts, oneHotUnitsAvailable, oneHotUnitsInProgress, metaTensor])
        stateTensor = np.concatenate([oneHotUnitCounts, oneHotUnitsAvailable, oneHotUnitsInProgress, metaTensor, foodTensor])
        return stateTensor, originalUnitCounts

    def combineStateAndGoal(self, stateTensor, goalTensor):
        assert goalTensor.shape == (self.num_units,)
        _stateTensor = stateTensor[0]
        unitCountsOrInProgress = stateTensor[1]
        remainingGoal = np.maximum(0, goalTensor - unitCountsOrInProgress)
        inputTensor = np.concatenate([_stateTensor, remainingGoal > 0, remainingGoal])
        assert inputTensor.shape == (self.tensor_input_size,)
        return inputTensor

    def createGoalTensor(self, goal):
        inputTensor = np.zeros(self.num_units, dtype=np.float32)
        for unit in goal:
            unitIndex = self.unit_lookup.unit_index_map[unit["type"]]
            inputTensor[unitIndex] += unit["count"]
            assert unit["count"] >= 0

        # s = inputTensor.sum()
        # if s > 0:
        #     inputTensor /= s
        return inputTensor

    def calculate_reward(self, t1, t2, goalTensor, deltaTime):
        s1unitCounts = t1[1]
        s2unitCounts = t2[1]
        assert s1unitCounts.shape == (self.num_units,)
        assert s2unitCounts.shape == (self.num_units,)

        # How many units were added
        deltaUnitCounts = np.maximum(0, s2unitCounts - s1unitCounts)
        # Total number of military units
        # numMilitaryUnits = (s1unitCounts * self.unit_lookup.military_units_mask).sum()
        # Number of units that we do want
        # desiredUnitCounts = goalTensor * np.maximum(numMilitaryUnits, 1)

        # falloff = 0.2
        # TODO: multiply by resource cost
        # scorePerUnit = goalTensor * np.minimum(1.0, np.exp((desiredUnitCounts - s1unitCounts)*falloff))
        scorePerUnit = np.zeros(self.num_units)
        scorePerUnit[4] = 1

        # Get a score if we added a unit of that type
        reward = (deltaUnitCounts * scorePerUnit).sum()

        assert(deltaTime >= 0)

        # Assume the reward happens right before s2.
        # If the build order involved satisfying some implicit constraints or maybe some waiting time
        # then the reward will be rewarded right at the end.
        # This makes it beneficial for the agent to learn to handle implicit dependencies by itself, but it can still fall back on them without too big of a loss.
        reward *= math.pow(self.gamma_per_second, deltaTime)
        return reward

    def determine_action(self, s1, s2, unit):
        pass

    def _add_goal_to_pool(self, goalTensor):
        self.goalPool.append(goalTensor)

        # Ensure the pool doesn't grow too large
        if len(self.goalPool) > 10000:
            self.goalPool[random.randrange(0, len(self.goalPool))] = self.goalPool[-1]
            self.goalPool.pop()

    def _sample_goals(self, n):
        return random.choices(self.goalPool, k=min(len(self.goalPool), n))

    def _create_transitions(self, states, tensor_states, goal_tensor, actions, failed):
        combined_tensor_states = [self.combineStateAndGoal(t, goal_tensor) for t in tensor_states]
        transitions = []
        rewards = []
        for i in range(len(tensor_states) - 1):
            s1 = states[i]
            s2 = states[i + 1]
            a1 = self.unit_lookup.unit_index_map[actions[i]]
            t1 = tensor_states[i]
            t2 = tensor_states[i + 1]
            ct1 = combined_tensor_states[i]
            ct2 = combined_tensor_states[i + 1]

            a2 = self.unit_lookup.unit_index_map[actions[i + 1]] if i + 1 < len(actions) else None
            deltaTime = s2["time"] - s1["time"]

            # reward = self.calculate_reward(t1, t2, goal_tensor, deltaTime)
            reward = 1 if a1 == 0 else 0
            terminal_state = a2 is None
            rewards.append(reward)

            # Skip terminal states at the moment
            # if terminal_state:
            # continue
            transition = Transition(state=ct1, action=a1, next_state=ct2, reward=reward, next_action=a2, deltaTime=deltaTime)
            transitions.append(transition)

        large_steps = False
        if large_steps:
            step_size = 5
            for i in range(len(tensor_states) - 2):
                k = min(len(tensor_states) - 1, i + step_size)
                s1 = states[i]
                s2 = states[k]
                a1 = self.unit_lookup.unit_index_map[actions[i]]
                t1 = combined_tensor_states[i]
                t2 = combined_tensor_states[k]
                a2 = self.unit_lookup.unit_index_map[actions[k]] if k < len(actions) else None

                reward = 0
                for j in range(i, k):
                    deltaTime = states[j]["time"] - s1["time"]
                    assert(deltaTime >= 0)
                    reward += math.pow(self.gamma_per_second, deltaTime) * rewards[j]

                deltaTime = s2["time"] - s1["time"]
                assert(deltaTime >= 0)

                terminal_state = a2 is None

                # Skip terminal states at the moment
                # if terminal_state:
                # continue

                # transition = Transition(state=t1, action=a1, next_state=t2, reward=reward, next_action=a2, deltaTime=deltaTime)
                # transitions.append(transition)

        return transitions

    def loadSession(self, s, memory, statistics):
        try:
            data = json.loads(s)
        except Exception as e:
            print("Failed to load session")
            print(e)
            return

        if "actions" not in data:
            print("Missing actions")
            return

        states = data["states"]
        actions = data["actions"]
        assert(len(actions) == len(states) - 1)

        goal_tensor = self.createGoalTensor(data["goal"])

        if goal_tensor.sum() == 0:
            # No goal, don't bother (usually the number of states is 1 anyway)
            return

        # goals = self._sample_goals(4)
        goals = []
        goals.append(goal_tensor)

        # self._add_goal_to_pool(goal_tensor)

        tensor_states_pre = [self.createState(s) for s in states]

        for goal in goals:
            total_reward = 0

            transitions = self._create_transitions(states, tensor_states_pre, goal, actions, data["failed"])
            total_reward = 0
            for transition in transitions:
                total_reward += transition.reward
                memory.push(transition)

            statistics.total_rewards.append(total_reward)
            statistics.durations.append(len(states))

# b = BuildOrderLoader(1)
# print("----")
# print(b.createState({ "units": [{ "type": 48, "totalCount": 2, "availableCount": 1 }], "unitsInProgress": []}))
# print("----")
# print(b.createState({ "units": [{ "type": 48, "totalCount": 5, "availableCount": 2312 }], "unitsInProgress": [{ "type": 48 }, { "type": 48 }, { "type": 48 }]}))
# exit(0)
