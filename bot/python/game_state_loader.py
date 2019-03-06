from build_order_loader import BuildOrderLoader, to_one_hot
from collections import namedtuple
import os
import PIL
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import torch

Trace = namedtuple('Trace', ['states', 'winner', 'replay_path', 'minimap_states', 'raw_unit_states', 'masks'])
MovementTrace = namedtuple('MovementTrace', ['states', 'movement', 'order_changed', 'replay_path', 'minimap_states',
                                             'raw_unit_states', 'raw_unit_coords', 'data_path', 'playerID'])
MovementTargetTrace = namedtuple('MovementTargetTrace', ['states', 'target_positions',
                                                         'unit_type_counts', 'replay_path', 'minimap_states', 'data_path', 'playerID', 'pathfinding_minimap', 'fraction_similar_orders'])

swap_winner_replays = [
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/231ccb5bdf905a4b7246a6800876fd03ca1eb7b0407ac9753c754e6675a63e44.SC2Replay",  # Other player rage-quits
    # Wrong player seems to quit for some reason
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/2d44e2f72f7147b3c9068a7417f9676e234dc250947405cf91017005319f99c6.SC2Replay",

    # Double check
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/16bcf92049063d9abb73f017519cd883aa66092e2d81656ab254bc024e815648.SC2Replay",
    # Wrong player seems to quit for some reason
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/061b34556dfcc36d7999e9a99faa4056a96057b1b50aa79afa90c433e25c5113.SC2Replay",
]

blacklisted_replays = [
    # Weird wall of planetary fortresses
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/144e15f5c4f42769c27a897680bbf0ebeebc8042250bdcd7a8ab7a9711e5e89c.SC2Replay",
    # SCV rush that failed... but the opponent still seems to give up?
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/0cf5d8f5cba9023cf4cb09f56711f51f50d2126bb781bfedcbfc2e3a1e45f7b8.SC2Replay",
    # Triggers an assertion for remaining time on a building
    "/home/arong/learning/sc2-voxelbot/replays/72b70403dd85296229108913b409740034ecab83bcdc1ecaa72c99067100240d.SC2Replay",
]

whitelisted_replays = [
    # Sneaky battlecruiser attack
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/046e6ab6b6d9cce5985493bff4f5fac7bc230395107f6e1dcceacd99c2248563.SC2Replay",
    # another sneaky battlecruiser attack
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/13f4e78a4a222a11954f16b684e2c72eb6d317e9615daeccc3d6af16d0e3d769.SC2Replay",
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/2874a258f9360311715aa66394060e165d318a6889af88cf47f015d39e095150.SC2Replay",  # Strange base-trade
]


def armyStrength(loader, state):
    count = 0
    for u in state["units"]:
        try:
            count += (
                loader.unit_lookup.military_units_mask[loader.unit_lookup.unit_index_map[u["type"]]] * 0.9 + 0.1) * u["totalCount"]
        except Exception:
            pass

    return count


def getInputTensorSize(loader: BuildOrderLoader):
    # return loader.get_tensor_size(loader.unit_lookup.all_unit_indices, False) + 2
    return 7 + 2 + loader.unit_lookup.num_upgrades


def start_unit(observationSession, playerID):
    units = observationSession["observations"]["rawUnits"][0]["units"]
    for u in units:
        if u["health"] > 100 and u["owner"] == playerID:
            return u

    assert False


def find_map_size(observationSession, playerID):
    start_unit1 = start_unit(observationSession, playerID)

    if False:
        start_unit2 = start_unit(observationSession, 1)
        mn_x = min(start_unit1["pos"]["x"], start_unit2["pos"]["x"])
        mn_y = min(start_unit1["pos"]["y"], start_unit2["pos"]["y"])
        mx_x = max(start_unit1["pos"]["x"], start_unit2["pos"]["x"])
        mx_y = max(start_unit1["pos"]["y"], start_unit2["pos"]["y"])

        s = max(mx_x / 2, mx_y / 2)
        mn_x = 0
        mn_y = 0
        mx_x = 2 * s
        mx_y = 2 * s
    else:
        mn_x, mn_y, mx_x, mx_y = findMapSizeFromMinimap(observationSession)
        margin = 4
        mn_x -= margin
        mn_y -= margin
        mx_x += margin
        mx_y += margin

    flipX = start_unit1["pos"]["x"] > (mn_x + mx_x) / 2
    flipY = start_unit1["pos"]["y"] > (mn_y + mx_y) / 2
    # print(((mn_x, mn_y), (mx_x, mx_y)), session["replayInfo"]["replay_path"])

    return ((mn_x, mn_y), (mx_x, mx_y), flipX, flipY)


def filter_matchup(session, loader):
    start_unit1 = start_unit({ "observations": session["observations"][0] }, 1)
    start_unit2 = start_unit({ "observations": session["observations"][1] }, 2)
    return start_unit1["unit_type"] in loader.unit_lookup.unit_index_map and start_unit2["unit_type"] in loader.unit_lookup.unit_index_map


# class SessionContext:
#     def __init__(session):
#         self.map_size = find_map_size(session)


# def loadState(context, observation1, observations2):
#     player1obs = playerObservationTensor(loader, observation1["selfStates"]) for x in
#     player2obs = playerObservationTensor(loader, observation2["selfStates"]) for x in
#     player1obs2 = playerObservationTensor2(loader, s, r, 1, map_size) for (s,r) in zip(observation1["selfStates"], observation1["rawUnits"])
#     player2obs2 = playerObservationTensor2(loader, s, r, 2, map_size) for (s,r) in zip(observation2["selfStates"], observation2["rawUnits"])

unitCountDistr = []


def filterSession(session, loader: BuildOrderLoader):
    if session["mmrs"][0] < 3000 and session["mmrs"][1] < 3000:
        print("Skipping due to low MMR")
        return False

    if session["winner"] == -1:
        print("Unknown winner, skipping replay")
        return False

    if session["winner"] == 0:
        print("Tie, skipping replay")
        return False

    if not filter_matchup(session, loader):
        print("Skipping due to not a supported matchup")
        return False

    gameDurationSeconds = session["replayInfo"]["duration_gameloops"] / 22.4

    # Skip very short games (very likely a player quitting due to some other reason)
    if gameDurationSeconds < 60:
        print("Skipping short game")
        return False

    if gameDurationSeconds > 40 * 60:
        print(f"Skipping long game ({gameDurationSeconds//60} minutes)")
        return False

    replay_path = session["replayInfo"]["replay_path"]

    if replay_path in blacklisted_replays:
        print("Skipping blacklisted replay")
        return False

    return True


def calculateWinner(session):
    # Convert from 1-indexed to 0-indexed
    winner = session["winner"] - 1
    if winner != 0 and winner != 1:
        print("Unexpected winner", winner, ". Skipping")
        return None

    assert winner == 0 or winner == 1
    replay_path = session["replayInfo"]["replay_path"]

    if replay_path in swap_winner_replays:
        print("Swapping winner for replay")
        winner = 1 - winner

    return winner


def loadSession(session, loader: BuildOrderLoader, memory, statistics):
    if not filterSession(session, loader):
        return

    replay_path = session["replayInfo"]["replay_path"]

    print("Processing")

    map_size = find_map_size(session)
    # player1obs = [playerObservationTensor(loader, x) for x in session["observations"][0]["selfStates"]]
    # player2obs = [playerObservationTensor(loader, x) for x in session["observations"][1]["selfStates"]]
    minimap_size = 5
    player1obs2 = [playerObservationTensor2(loader, s, r, 1, map_size, minimap_size) for (s, r) in zip(
        session["observations"][0]["selfStates"], session["observations"][0]["rawUnits"])]
    player2obs2 = [playerObservationTensor2(loader, s, r, 2, map_size, minimap_size) for (s, r) in zip(
        session["observations"][1]["selfStates"], session["observations"][1]["rawUnits"])]
    globalObs = [globalObservationTensor(loader, map_size, a, b, False) for (a, b) in zip(
        session["observations"][0]["rawUnits"], session["observations"][1]["rawUnits"])]
    mirroredGlobalObs = [globalObservationTensor(loader, map_size, a, b, True) for (a, b) in zip(
        session["observations"][0]["rawUnits"], session["observations"][1]["rawUnits"])]
    rawUnitTensors = [a for (a, _) in globalObs]
    maskTensors = [b for (_, b) in globalObs]

    mirroredRawUnitTensors = [a for (a, _) in mirroredGlobalObs]
    mirroredMaskTensors = [b for (_, b) in mirroredGlobalObs]
    # global unitCountDistr
    # unitCounts = [sum(x["totalCount"] for x in a["units"]) + sum(x["totalCount"] for x in b["units"]) for (a,b) in zip(session["observations"][0]["selfStates"], session["observations"][1]["selfStates"])]
    # unitCountDistr += [sum(x["totalCount"] for x in a["units"]) + sum(x["totalCount"] for x in b["units"]) for (a,b) in zip(session["observations"][0]["selfStates"], session["observations"][1]["selfStates"])]
    # plt.clf()
    # plt.hist(unitCountDistr, bins=20)
    # plt.pause(0.001)

    # player1obs = [np.concatenate([a,b]) for (a,b) in zip(player1obs, player1obs2)]
    # player2obs = [np.concatenate([a,b]) for (a,b) in zip(player2obs, player2obs2)]
    player1obs = player1obs2
    player2obs = player2obs2

    states = [np.stack([a, b]) for ((a, _), (b, _)) in zip(player1obs, player2obs)]
    mirrored_states = [np.stack([b, a]) for ((a, _), (b, _)) in zip(player1obs, player2obs)]
    # minimap_states = [np.stack([a, b]) for ((_,a),(_,b)) in zip(player1obs, player2obs)]
    # mirrored_minimap_states = [np.stack([b, a]) for ((_,a),(_,b)) in zip(player1obs, player2obs)]

    # winner = 0 | 1
    winner = calculateWinner(session)
    if winner is None:
        return

    p1s = armyStrength(loader, session["observations"][winner]["selfStates"][-1])
    p2s = armyStrength(loader, session["observations"][1 - winner]["selfStates"][-1])

    if p1s > p2s:
        pass
    else:
        print("Not consistent ", p1s, p2s)
        if p1s < p2s / 3.0:
            if replay_path in whitelisted_replays:
                print("Very not consistent, however this replay is whitelisted")
            else:
                print(f"Very not consistent ({p1s} < {p2s})", replay_path)
                print("Skipping")
                return

    trace = Trace(states=states, winner=winner, replay_path=replay_path,
                  minimap_states=None, raw_unit_states=rawUnitTensors, masks=maskTensors)
    mirror = Trace(states=mirrored_states, winner=1 - winner, replay_path=replay_path,
                   minimap_states=None, raw_unit_states=mirroredRawUnitTensors, masks=mirroredMaskTensors)
    memory.push(trace)
    memory.push(mirror)


def playerObservationTensor(loader: BuildOrderLoader, state):
    tensor = loader.createState(state, unitIndices=loader.unit_lookup.all_unit_indices)
    return tensor[0]


def transform_coord(coord, map_size, mirror):
    mn, mx, flipX, flipY = map_size
    normalized_x = (coord["x"] - mn[0]) / (mx[0] - mn[0])
    normalized_y = (coord["y"] - mn[1]) / (mx[1] - mn[1])

    if flipX != mirror:
        normalized_x = 1 - normalized_x
    if flipY != mirror:
        normalized_y = 1 - normalized_y

    return normalized_x, normalized_y


def inverse_transform_coord(normalized_coord, map_size, mirror):
    mn, mx, flipX, flipY = map_size
    normalized_x, normalized_y = normalized_coord
    if flipX != mirror:
        normalized_x = 1 - normalized_x
    if flipY != mirror:
        normalized_y = 1 - normalized_y

    x = normalized_x * (mx[0] - mn[0]) + mn[0]
    y = normalized_y * (mx[1] - mn[1]) + mn[1]

    return x, y

def transform_coord_minimap(coord, map_size, scale, mirror):
    normalized_x, normalized_y = transform_coord(coord, map_size, mirror)

    normalized_x *= scale
    normalized_y *= scale
    return (min(scale - 1, max(0, int(round(normalized_x)))), min(scale - 1, max(0, int(round(normalized_y)))))


def inverse_transform_coord_minimap(coord, map_size, scale, mirror):
    coord = (coord[0]/scale, coord[1]/scale)
    return inverse_transform_coord(coord, map_size, mirror)


def playerObservationTensor2(loader: BuildOrderLoader, state, raw_units, playerID, map_size, minimap_size, mirror):
    units = raw_units["units"]
    total_army_health = 0
    total_army_health_fraction = 0.000001
    total_army_health_fraction_weight = 0.000001

    minimap_health = np.zeros((minimap_size, minimap_size, loader.num_units), dtype=np.float32)
    minimap_count = np.zeros((minimap_size, minimap_size, loader.num_units), dtype=np.float32)
    flying_health = np.zeros((minimap_size, minimap_size, loader.num_units), dtype=np.float32)

    def isArmy(unit_type):
        if unit_type in loader.unit_lookup.unit_index_map:
            return loader.unit_lookup.military_units_mask[loader.unit_lookup.unit_index_map[unit_type]] > 0
        else:
            return False

    for unit in units:
        if not unit["is_alive"]:
            continue

        if unit["owner"] == playerID:
            unit_type = unit["unit_type"]
            if unit_type in loader.unit_lookup.unit_index_map:
                coord = transform_coord_minimap(unit["pos"], map_size, minimap_size, mirror)
                index = loader.unit_lookup.unit_index_map[unit_type]
                minimap_count[coord[0], coord[1], index] += 1
                if unit["is_flying"]:
                    flying_health[coord[0], coord[1], index] += (unit["health"] + unit["shield"])  # / (unit["health_max"] + unit["shield_max"])
                else:
                    minimap_health[coord[0], coord[1], index] += (unit["health"] + unit["shield"])  # / (unit["health_max"] + unit["shield_max"])

            if isArmy(unit_type):
                total_army_health += unit["health"] + unit["shield"]

                total_army_health_fraction += unit["health"] + unit["shield"]
                total_army_health_fraction_weight += (unit["health_max"] + unit["shield_max"])

    # total_minimap_health = np.sum(minimap_health, axis=2)
    # print(total_minimap_health)
    # fig = plt.figure(2)
    # fig.clf()
    # plt.imshow(total_minimap_health, interpolation='nearest', cmap=plt.cm.inferno)
    # plt.colorbar()
    # plt.show()
    # plt.pause(0.001)

    minimap_health = minimap_health.sum(axis=2)
    minimap_health = np.stack([minimap_health, minimap_count.sum(axis=2), flying_health.sum(axis=2)], axis=-1)

    total_army_health_fraction /= total_army_health_fraction_weight

    # sorted_units = np.argsort(minimap_count, axis=2)[:,:,-5:]
    # minimap_count = np.take_along_axis(minimap_count, sorted_units, axis=2)
    # Some metadata, the data is normalized to approximately 1
    metaTensor = np.zeros(7, dtype=np.float32)
    metaTensor[0] = state["minerals"] / 100
    metaTensor[1] = state["vespene"] / 100
    metaTensor[2] = state["mineralsPerSecond"] / 10
    metaTensor[3] = state["vespenePerSecond"] / 10
    metaTensor[4] = state["highYieldMineralSlots"] / 10
    metaTensor[5] = state["lowYieldMineralSlots"] / 10
    metaTensor[6] = minimap_count.sum() / 50

    upgradeTensor = np.zeros(loader.unit_lookup.num_upgrades, dtype=np.float32)
    for u in state["upgrades"]:
        if u in loader.unit_lookup.upgrade_index_map:
            upgradeTensor[loader.unit_lookup.upgrade_index_map[u]] = 1

    # harvesters = [45, 104, 116, 84]
    # for h_type in harvesters:
    #     metaTensor[6] += unitCounts[self.unit_lookup.unit_index_map[h_type]] / 10  # SCV+Drone+Probe count
    # foodTensor = to_one_hot(np.array([state["foodAvailable"]]), 8)

    armyTensor = np.array([total_army_health / 1000, total_army_health_fraction], dtype=np.float32)
    return torch.tensor(np.concatenate([metaTensor, upgradeTensor, armyTensor])), torch.tensor(minimap_health)


def minimapLayers(loader: BuildOrderLoader, raw_units, playerID, map_size, minimap_size, mirror):
    units = raw_units["units"]
    total_army_health = 0
    total_army_health_fraction = 0.000001
    total_army_health_fraction_weight = 0.000001

    minimap_health = np.zeros((minimap_size, minimap_size, loader.num_units), dtype=np.float32)
    minimap_count = np.zeros((minimap_size, minimap_size, loader.num_units), dtype=np.float32)
    flying_health = np.zeros((minimap_size, minimap_size, loader.num_units), dtype=np.float32)
    minimap_health_buildings = np.zeros((minimap_size, minimap_size, loader.num_units), dtype=np.float32)
    minimap_health_workers = np.zeros((minimap_size, minimap_size, loader.num_units), dtype=np.float32)

    def isArmy(unit_type):
        if unit_type in loader.unit_lookup.unit_index_map:
            return loader.unit_lookup.military_units_mask[loader.unit_lookup.unit_index_map[unit_type]] > 0
        else:
            return False

    for unit in units:
        if not unit["is_alive"]:
            continue

        if unit["owner"] == playerID:
            unit_type = unit["unit_type"]
            if unit_type in loader.unit_lookup.unit_index_map:
                coord = transform_coord_minimap(unit["pos"], map_size, minimap_size, mirror)
                index = loader.unit_lookup.unit_index_map[unit_type]
                minimap_count[coord[0], coord[1], index] += 1
                max_health = max(unit["health_max"] + unit["shield_max"], 1)
                if isArmy(unit_type):
                    if unit["is_flying"]:
                        flying_health[coord[0], coord[1], index] += (unit["health"] + unit["shield"])  # / (unit["health_max"] + unit["shield_max"])
                    else:
                        minimap_health[coord[0], coord[1], index] += (unit["health"] + unit["shield"])  # / (unit["health_max"] + unit["shield_max"])

                if not isMovableUnit(unit, loader):
                    minimap_health_buildings[coord[0], coord[1], index] += (unit["health"] + unit["shield"]) / max_health

                if unit_type in loader.unit_lookup.workerUnitTypes:
                    minimap_health_workers[coord[0], coord[1], index] += (unit["health"] + unit["shield"]) / max_health

    # total_minimap_health = np.sum(minimap_health, axis=2)
    # print(total_minimap_health)
    # fig = plt.figure(2)
    # fig.clf()
    # plt.imshow(total_minimap_health, interpolation='nearest', cmap=plt.cm.inferno)
    # plt.colorbar()
    # plt.show()
    # plt.pause(0.001)

    minimap_health = minimap_health.sum(axis=2)
    minimap = torch.tensor(np.stack([
        minimap_health,
        minimap_count.sum(axis=2),
        flying_health.sum(axis=2),
        minimap_health_buildings.sum(axis=2),
        minimap_health_workers.sum(axis=2)
    ]))
    return minimap


def globalObservationTensor(loader: BuildOrderLoader, map_size, player1_raw_units, player2_raw_units, mirror):
    def transform_coord(coord):
        mn, mx, flipX, flipY = map_size
        normalized_x = (coord["x"] - mn[0]) / (mx[0] - mn[0])
        normalized_y = (coord["y"] - mn[1]) / (mx[1] - mn[1])

        if flipX != mirror:
            normalized_x = 1 - normalized_x
        if flipY != mirror:
            normalized_y = 1 - normalized_y
        return (normalized_x, normalized_y)

    def unitObservation(unit, result):
        assert unit["is_alive"]
        assert unit["display_type"] == 1
        assert not unit["is_blip"]
        assert result.shape == (11,)
        health = (unit["health"] + unit["shield"]) / (unit["health_max"] + unit["shield_max"])
        result[0] = loader.unit_lookup.unit_index_map[unit["unit_type"]]
        coords = transform_coord(unit["pos"])
        result[1] = math.cos(coords[0] * math.pi * 1.0)
        result[2] = math.sin(coords[0] * math.pi * 1.0)
        result[3] = math.cos(coords[1] * math.pi * 1.0)
        result[4] = math.sin(coords[1] * math.pi * 1.0)
        result[5] = math.cos(coords[0] * math.pi * 2.0)
        result[6] = math.sin(coords[0] * math.pi * 2.0)
        result[7] = math.cos(coords[1] * math.pi * 2.0)
        result[8] = math.sin(coords[1] * math.pi * 2.0)
        result[9] = unit["owner"] * 2 - 3  # Convert owner (1|2) to -1 or 1
        if mirror:
            result[9] *= -1
        assert result[9] == -1 or result[9] == 1
        result[10] = health
        return result

    num_input_units = 200

    valid_units = [[], []]
    combined_units = []
    for unit in player1_raw_units["units"]:
        if unit["owner"] == 1 and unit["unit_type"] in loader.unit_lookup.unit_index_map:
            valid_units[0].append(unit)
    for unit in player2_raw_units["units"]:
        if unit["owner"] == 2 and unit["unit_type"] in loader.unit_lookup.unit_index_map:
            valid_units[1].append(unit)

    for us in valid_units:
        # Remove units with the lowest health if needed
        # (units at the end of the list will be used first)
        us.sort(key=lambda unit: unit["health"] + unit["shield"])

    # plt.clf()
    # plt.scatter(x=[transform_coord(u["pos"])[0] for u in valid_units[0]], y=[transform_coord(u["pos"])[1] for u in valid_units[0]], c="#FF0000")
    # plt.scatter(x=[transform_coord(u["pos"])[0] for u in valid_units[1]], y=[transform_coord(u["pos"])[1] for u in valid_units[1]], c="#0000FF")

    changed = True
    while changed:
        changed = False
        for i in range(2):
            if len(combined_units) < num_input_units and len(valid_units[i]) > 0:
                changed = True
                combined_units.append(valid_units[i].pop())

    # plt.imshow(total_minimap_health, interpolation='nearest', cmap=plt.cm.inferno)
    # plt.colorbar()
    # plt.clf()
    # colors = ["#FF0000" if u["owner"] == 1 else "#0000FF" for u in combined_units]
    # xs = list(range(len(combined_units)))
    # plt.bar(xs, [u["health"] + u["shield"] for u in combined_units], color=colors)
    # plt.xlim([0, num_input_units])
    # # plt.show()
    plt.pause(0.001)

    allUnits = np.zeros((num_input_units, 11))

    mask = []
    for i, unit in enumerate(combined_units):
        unitObservation(unit, result=allUnits[i])

    mask = np.zeros(num_input_units)
    mask[:len(combined_units)] = 1

    return allUnits, mask


def loadSessionMovement(session, loader: BuildOrderLoader, store_fn, statistics):
    if not filterSession(session, loader):
        return

    # winner = 0 | 1
    winner = calculateWinner(session)
    if winner is None:
        return

    # Map unit -> last known position
    # For every time
    #   If unit.pos is far from its last known position. Mark the unit as moved

    playerID = winner + 1
    observationSession = {
        "observations": session["observations"][playerID - 1],
        "gameInfo": session["gameInfo"],
        "replayInfo": session["replayInfo"]
    }
    trace = loadSessionMovement2(observationSession, playerID, loader, True, session["data_path"])
    store_fn(trace)


def loadSessionMovement2(observationSession, playerID, loader: BuildOrderLoader, extract_movement, data_path):
    playerIndex = playerID - 1
    opponentPlayerID = 3 - playerID
    map_size = find_map_size(observationSession, playerID)

    replay_path = observationSession["replayInfo"]["replay_path"]

    # Coordinates are normalized so that [playerID] is always in the lower left corner.
    mirror = False

    selfStates = observationSession["observations"]["selfStates"]
    rawUnits = observationSession["observations"]["rawUnits"]

    player1movement = extractMovement(rawUnits, playerID, map_size, loader, mirror) if extract_movement else None
    player1orderChanged = extractOrderChanged(rawUnits, playerID, map_size, loader, mirror) if extract_movement else None
    # player2movement = extractMovement(session["observations"][1]["rawUnits"], 1, map_size, loader)

    minimap_size = 10
    player1obs2 = [playerObservationTensor2(loader, s, r, playerID, map_size, minimap_size, mirror)
                   for (s, r) in zip(selfStates, rawUnits)]
    # player2obs2 = [playerObservationTensor2(loader, s, r, 2, map_size, minimap_size) for (s, r) in zip(selfStates[1], rawUnits[1])]

    player1units = [movementUnitStates(r, loader, playerID, map_size, minimap_size, mirror) for (s, r) in zip(selfStates, rawUnits)]
    # player2units = [movementUnitStates(r, loader, 2, map_size, minimap_size, False) for (s, r) in zip(selfStates[1], rawUnits[1])]

    player1minimap1 = [minimapLayers(loader, r, playerID, map_size, minimap_size, mirror) for r in rawUnits]
    player1minimap2 = [minimapLayers(loader, r, opponentPlayerID, map_size, minimap_size, mirror) for r in rawUnits]
    player1minimap = torch.stack([torch.cat((m1, m2), dim=0) for (m1, m2) in zip(player1minimap1, player1minimap2)])
    # player2minimap2 = minimapLayers(loader, session["observations"][0]["rawUnits"], 1, map_size, minimap_size)

    return MovementTrace(
        states=torch.stack([x[0] for x in player1obs2]),
        movement=player1movement,
        order_changed=player1orderChanged,
        minimap_states=player1minimap,
        replay_path=replay_path,
        raw_unit_states=[x[0] for x in player1units],
        raw_unit_coords=[x[1] for x in player1units],
        data_path=data_path,
        playerID=playerID,
    )


def loadSessionMovementTarget(session, loader: BuildOrderLoader, store_fn, statistics):
    if not filterSession(session, loader):
        return

    map_name = session["gameInfo"]["map_name"]
    if map_name == "Stasis LE":
        print(f"Skipping blacklisted map {map_name}")
        return

    # winner = 0 | 1
    winner = calculateWinner(session)
    if winner is None:
        return

    # Map unit -> last known position
    # For every time
    #   If unit.pos is far from its last known position. Mark the unit as moved

    playerID = winner + 1
    observationSession = {
        "observations": session["observations"][playerID-1],
        "gameInfo": session["gameInfo"],
        "replayInfo": session["replayInfo"]
    }
    res = loadSessionMovementTarget2(observationSession, playerID, loader, 'random', session["data_path"])
    if res is not None:
        store_fn(res)


def loadSessionMovementTarget2(observationSession, playerID, loader: BuildOrderLoader, unit_tag_mask, data_path):
    ''' unit_tag_mask is either 'random' or a set of unit tags'''

    replay_path = observationSession["replayInfo"]["replay_path"]
    playerIndex = playerID - 1
    opponentPlayerID = 3 - playerID

    map_size = find_map_size(observationSession, playerID)

    # Coordinates are normalized so that player [playerID] is always in the lower left corner.
    mirror = False

    selfStates = observationSession["observations"]["selfStates"]
    rawUnits = observationSession["observations"]["rawUnits"]
    minimap_size = 14

    # In timesteps, so Nx5 seconds
    # Note: if unit_tag_mask is provided then we don't use lookahead for anything, so it doesn't constrain the times
    lookaheadTime = 4 if unit_tag_mask == 'random' else 0
    max_time = len(rawUnits) - lookaheadTime

    if max_time <= 0:
        print("Skipping game with too few samples")
        return

    player1obs2 = [
        playerObservationTensor2(loader, s, r, playerID, map_size, minimap_size, mirror)
        for (s, r) in zip(selfStates[:max_time], rawUnits[:max_time])
    ]

    player1minimap1 = [minimapLayers(loader, r, playerID, map_size, minimap_size, mirror) for r in rawUnits[:max_time]]
    player1minimap2 = [minimapLayers(loader, r, opponentPlayerID, map_size, minimap_size, mirror) for r in rawUnits[:max_time]]
    player1minimap = torch.stack([torch.cat((m1, m2), dim=0) for (m1, m2) in zip(player1minimap1, player1minimap2)])

    if unit_tag_mask == "random":
        player1movementMinimap, unit_type_counts, target_positions, fractionSimilarOrders = sampleMovementGroupOrder(rawUnits, playerID, map_size, loader, minimap_size, mirror, lookaheadTime)
    else:
        target_positions = None
        assert len(rawUnits) == 1
        player1movementMinimap, unit_type_counts = movementGroupFeatures([u for u in rawUnits[0]["units"] if u["tag"] in unit_tag_mask], loader, map_size, minimap_size, mirror)
        player1movementMinimap = torch.tensor(player1movementMinimap).unsqueeze(0)
        unit_type_counts = torch.tensor(unit_type_counts).unsqueeze(0)

    player1minimap = torch.cat([player1minimap, player1movementMinimap], dim=1)

    return MovementTargetTrace(
        states=torch.stack([x[0] for x in player1obs2]),
        minimap_states=player1minimap,
        replay_path=replay_path,
        target_positions=target_positions,
        unit_type_counts=unit_type_counts,
        fraction_similar_orders=fractionSimilarOrders,
        data_path=data_path,
        playerID=playerID,
        pathfinding_minimap=loadPathfindingMinimap(observationSession, map_size, mirror)
    )

    # plt.clf()
    # plt.imshow(res1.pathfinding_minimap.transpose(0, 1), origin='lower')
    # s = start_unit(session, playerIndex)
    # coords = np.array([transform_coord(u["pos"], map_size, mirror) for u in rawUnits[0][1]["units"]])
    # p = transform_coord(s["pos"], map_size, mirror)
    # plt.scatter(coords[:, 0] * 168, coords[:, 1] * 168, c="#00FF00", marker='.')
    # plt.scatter([p[0] * 168], [p[1] * 168], c="#FF0000")
    # plt.pause(1)


cached_map_sizes = {}


def findMapSizeFromMinimap(session):
    '''
    Returns [mn_x, mn_y, mx_x, mx_y]
    '''
    def bbox1(img):
        a = np.nonzero(img)
        bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
        return bbox

    name = session["replayInfo"]["map_name"]
    filepath = f"training_data/maps/1/{name}.pickle"
    minimap = None

    if filepath in cached_pathfinding_minimaps:
        return cached_map_sizes[filepath]

    if not os.path.exists(filepath):
        raise Exception(f"Could not find minimap at {filepath} (replay={session['replayInfo']['replay_path']})")

    v = torch.load(filepath)
    minimap = v[2]
    res = bbox1(minimap.numpy())
    cached_map_sizes[filepath] = res
    return res


cached_pathfinding_minimaps = {}

def loadPathfindingMinimap(session, map_size, mirror):
    name = session["replayInfo"]["map_name"]
    filepath = f"training_data/maps/1/{name}.pickle"
    minimap = None
    mn, mx, flipX, flipY = map_size

    if filepath in cached_pathfinding_minimaps:
        minimap = cached_pathfinding_minimaps[filepath]

    if minimap is None and os.path.exists(filepath):
        v = torch.load(filepath)
        minimap = v[2]

        c1 = torchvision.transforms.ToPILImage()
        c2 = torchvision.transforms.Resize(size=(168, 168), interpolation=PIL.Image.NEAREST)
        c3 = torchvision.transforms.ToTensor()
        minimap = c1(minimap.unsqueeze(0))

        # plt.clf()
        # plt.imshow(c3(minimap).squeeze(0))
        # plt.plot([0, 168], [mn[1], mn[1]])
        # plt.plot([0, 168], [mx[1], mx[1]])
        # plt.plot([mn[0], mn[0]], [0, 168])
        # plt.plot([mx[0], mx[0]], [0, 168])
        # plt.pause(2)
        minimap = torchvision.transforms.functional.crop(minimap, int(round(mn[0])), int(round(mn[1])), int(round(mx[0] - mn[0])), int(round(mx[1] - mn[1])))
        minimap = c2(minimap)
        minimap = c3(minimap).squeeze(0)  # .transpose(0, 1)
        cached_pathfinding_minimaps[filepath] = minimap

    if minimap is not None:
        flipDims = []
        if flipX != mirror:
            flipDims.append(0)

        if flipY != mirror:
            flipDims.append(1)

        return torch.flip(minimap, flipDims)
    else:
        raise Exception(f"Could not find minimap at {filepath} (replay={session['replayInfo']['replay_path']})")


def squaredDistance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def sampleMovementGroup(rawUnits, playerID, map_size, loader, minimap_size, mirror, lookaheadTime):
    minimap_layers = []
    unit_types = []
    target_positions = []
    fractionSimilarOrders = []

    for i, state in enumerate(rawUnits[:-lookaheadTime]):
        valid_unit_tags = set()
        for unit in state["units"]:
            is_worker = unit["unit_type"] in loader.unit_lookup.workerUnitTypes
            if unit["owner"] == playerID and isMovableUnit(unit, loader) and unit["unit_type"] in loader.unit_lookup.unit_index_map and not is_worker:
                valid_unit_tags.add(unit["tag"])

        has_any_units = np.zeros((minimap_size, minimap_size))
        for future_unit in rawUnits[i + lookaheadTime]["units"]:
            if future_unit["tag"] in valid_unit_tags:
                coord = transform_coord_minimap(future_unit["pos"], map_size, minimap_size, mirror)
                has_any_units[coord[0], coord[1]] = 1

        target_position = np.zeros((minimap_size, minimap_size), dtype=np.float32)
        filtered_tags = set()

        if has_any_units.sum() > 0:
            has_any_units = has_any_units.flatten() / has_any_units.sum()
            cell_index = np.random.choice(has_any_units.shape[0], p=has_any_units)
            cell_x = cell_index // minimap_size
            cell_y = cell_index % minimap_size

            for future_unit in rawUnits[i + lookaheadTime]["units"]:
                if future_unit["tag"] in valid_unit_tags:
                    coord = transform_coord_minimap(future_unit["pos"], map_size, minimap_size, mirror)
                    if coord[0] == cell_x and coord[1] == cell_y:
                        filtered_tags.add(future_unit["tag"])

            assert len(filtered_tags) > 0

            target_position[cell_x, cell_y] = 1

        units = [u for u in state["units"] if u["tag"] in filtered_tags]
        unitsFuture = [u for u in rawUnits[i + 1]["units"] if u["tag"] in filtered_tags]
        unitsComplement = [u for u in rawUnits[i + 0]["units"] if u["tag"] not in filtered_tags]
        layers, unit_type_counts, fractionSimilar = movementGroupFeatures(units, unitsFuture, unitsComplement, loader, map_size, minimap_size, mirror)
        minimap_layers.append(layers)
        unit_types.append(unit_type_counts)
        target_positions.append(target_position)
        fractionSimilarOrders.append(fractionSimilar)

    minimap_layers = torch.tensor(np.stack(minimap_layers))
    unit_types = torch.tensor(np.stack(unit_types))
    target_positions = torch.tensor(np.stack(target_positions))
    fractionSimilarOrders = torch.tensor(fractionSimilarOrders, dtype=torch.float32)
    return minimap_layers, unit_types, target_positions, fractionSimilarOrders


def sampleMovementGroupOrder(rawUnits, playerID, map_size, loader, minimap_size, mirror, lookaheadTime):
    '''
    Extract unit groups which have the same destination one frame in the future (but not necessarily this frame)
    '''
    minimap_layers = []
    unit_types = []
    target_positions = []
    fractionSimilarOrders = []

    for i, state in enumerate(rawUnits[:-lookaheadTime]):
        has_any_units = np.zeros((minimap_size, minimap_size))
        existing_unit_tags = [u["tag"] for u in state["units"]]
        valid_units = []
        for unit in rawUnits[i + 1]["units"]:
            is_worker = unit["unit_type"] in loader.unit_lookup.workerUnitTypes
            unit_exists = unit["tag"] in existing_unit_tags
            known_type = unit["unit_type"] in loader.unit_lookup.unit_index_map
            if (not is_worker) and unit["owner"] == playerID and isMovableUnit(unit, loader) and known_type and unit_exists:
                coord = transform_coord_minimap(unitDestination(unit), map_size, minimap_size, mirror)
                has_any_units[coord[0], coord[1]] = 1
                valid_units.append(unit)

        target_position = np.zeros((minimap_size, minimap_size), dtype=np.float32)
        filtered_unit_tags = set()

        if has_any_units.sum() > 0:
            has_any_units = has_any_units.flatten() / has_any_units.sum()
            cell_index = np.random.choice(has_any_units.shape[0], p=has_any_units)
            cell_x = cell_index // minimap_size
            cell_y = cell_index % minimap_size

            # TODO: Make sure they are not far from each other originally
            for unit in valid_units:
                coord = transform_coord_minimap(unitDestination(unit), map_size, minimap_size, mirror)
                if coord[0] == cell_x and coord[1] == cell_y:
                    filtered_unit_tags.add(unit["tag"])

            assert len(filtered_unit_tags) > 0

            target_position[cell_x, cell_y] = 1

        units = [u for u in state["units"] if u["tag"] in filtered_unit_tags and u["unit_type"] in loader.unit_lookup.unit_index_map]

        unitsFuture = [u for u in rawUnits[i + 1]["units"] if u["tag"] in filtered_unit_tags]
        # Note: includes workers
        unitsComplement = [u for u in state["units"] if u["owner"] == playerID and isMovableUnit(u, loader) and u["tag"] not in filtered_unit_tags]
        layers, unit_type_counts, fractionSimilar = movementGroupFeatures(units, unitsFuture, unitsComplement, loader, map_size, minimap_size, mirror)
        minimap_layers.append(layers)
        unit_types.append(unit_type_counts)
        target_positions.append(target_position)
        fractionSimilarOrders.append(fractionSimilar)

    minimap_layers = torch.tensor(np.stack(minimap_layers))
    unit_types = torch.tensor(np.stack(unit_types))
    target_positions = torch.tensor(np.stack(target_positions))
    fractionSimilarOrders = torch.tensor(fractionSimilarOrders, dtype=torch.float32)
    return minimap_layers, unit_types, target_positions, fractionSimilarOrders


def unitOrderDestination(unit):
    if len(unit["orders"]) == 0:
        return None

    order = unit["orders"][0]
    target_pos = order["target_pos"]
    if target_pos is not None and (target_pos["x"] != 0 or target_pos["y"] != 0):
        return target_pos

    # TODO: map target_unit_tag to position
    return None


def unitDestination(unit):
    orderDest = unitOrderDestination(unit)
    if orderDest is not None:
        return orderDest

    return unit["pos"]


def similarOrders(unit1, unit2):
    distanceThreshold = 6
    # Special case
    # If the future unit has no orders, but the current unit has an order
    # that has the future position of the unit as the destination, we say that they are similar.
    # It is likely that the unit continued with the order until it reached its destination.
    if len(unit1["orders"]) > 0 and len(unit2["orders"]) == 0:
        dest1 = unit1["orders"][0]["target_pos"]
        if dest1 is not None:
            if squaredDistance((dest1["x"], dest1["y"]), (unit2["pos"]["x"], unit2["pos"]["y"])) <= distanceThreshold * distanceThreshold:
                return True

    if (len(unit1["orders"]) > 0) != (len(unit2["orders"]) > 0):
        return False

    # Both are standing still
    if (len(unit1["orders"]) == 0) and (len(unit2["orders"]) == 0):
        return True

    order1 = unit1["orders"][0]
    order2 = unit2["orders"][0]

    if order1["ability_id"] != order2["ability_id"]:
        return False

    if order1["target_unit_tag"] != order2["target_unit_tag"]:
        return False

    dest1 = order1["target_pos"]
    dest2 = order2["target_pos"]

    if (dest1 is not None) != (dest2 is not None):
        return False

    if squaredDistance((dest1["x"], dest1["y"]), (dest2["x"], dest2["y"])) > distanceThreshold * distanceThreshold:
        return False

    return True


def movementGroupFeatures(units, unitsFuture, unitsComplement, loader, map_size, minimap_size, mirror):
    has_unit = np.zeros((minimap_size, minimap_size), dtype=np.float32)
    normalized_unit_counts = np.zeros((minimap_size, minimap_size), dtype=np.float32)
    unit_destinations = np.zeros((minimap_size, minimap_size), dtype=np.float32)
    unit_type_counts = np.zeros((loader.unit_lookup.num_units), dtype=np.float32)
    normalized_unit_counts = np.zeros((minimap_size, minimap_size), dtype=np.float32)
    unit_destinations_complement = np.zeros((minimap_size, minimap_size), dtype=np.float32)
    tag2futureUnit = {u["tag"]: u for u in unitsFuture}

    similar = 0
    totalOrders = 0

    for unit in units:
        coord = transform_coord_minimap(unit["pos"], map_size, minimap_size, mirror)
        has_unit[coord[0], coord[1]] = 1
        normalized_unit_counts[coord[0], coord[1]] += 1
        unit_type_counts[loader.unit_lookup.unit_index_map[unit["unit_type"]]] += 1

        target = unitDestination(unit)
        coord = transform_coord_minimap(target, map_size, minimap_size, mirror)
        unit_destinations[coord[0], coord[1]] += 1

        futureUnit = tag2futureUnit[unit["tag"]] if unit["tag"] in tag2futureUnit else None
        if futureUnit is not None:
            if similarOrders(unit, futureUnit):
                similar += 1

            totalOrders += 1

    if totalOrders > 0:
        fractionSimilar = similar / totalOrders
    else:
        # Assume they just continued doing what they were doing
        # Unfortunately they died
        fractionSimilar = 1

    for unit in unitsComplement:
        target = unitDestination(unit)
        coord = transform_coord_minimap(target, map_size, minimap_size, mirror)
        unit_destinations_complement[coord[0], coord[1]] += 1

    if len(units) > 0:
        normalized_unit_counts /= normalized_unit_counts.sum()
        unit_type_counts /= unit_type_counts.sum()

    if len(unitsComplement) > 0:
        unit_destinations_complement /= unit_destinations_complement.sum()

    if len(units) > 0:
        unit_destinations /= unit_destinations.sum()

    minimap_layers = np.stack([has_unit, normalized_unit_counts, unit_destinations, unit_destinations_complement])
    return minimap_layers, unit_type_counts, fractionSimilar


def extractMovement(rawUnits, playerID, map_size, loader, mirror):
    def transform_coord(coord):
        mn, mx, flipX, flipY = map_size
        normalized_x = (coord["x"] - mn[0]) / (mx[0] - mn[0])
        normalized_y = (coord["y"] - mn[1]) / (mx[1] - mn[1])

        if flipX != mirror:
            normalized_x = 1 - normalized_x
        if flipY != mirror:
            normalized_y = 1 - normalized_y
        return (normalized_x, normalized_y)

    # ["observations"][0]["rawUnits"]
    lastKnownPositions = {}
    didMove = []
    moves = []
    for i in range(len(rawUnits)):
        moves.append(set())

    movementThreshold = 6

    for i, state in enumerate(rawUnits):
        units = state["units"]
        for unit in units:
            if unit["owner"] != playerID:
                continue

            tag = unit["tag"]
            newPos = (unit["pos"]["x"], unit["pos"]["y"])
            if tag not in lastKnownPositions:
                lastKnownPositions[tag] = newPos
            else:
                oldPos = lastKnownPositions[tag]
                if squaredDistance(oldPos, newPos) > movementThreshold * movementThreshold:
                    # Register movement
                    moves[i - 1].add(tag)
                    lastKnownPositions[tag] = newPos

    for i in range(len(rawUnits)):
        playerUnits = [u for u in rawUnits[i]["units"] if u["owner"] == playerID and isMovableUnit(u, loader)]
        numPlayerUnits = len(playerUnits)
        didMoveThisFrame = torch.zeros(numPlayerUnits, dtype=torch.long)
        for j, u in enumerate(playerUnits):
            if u["tag"] in moves[i]:
                didMoveThisFrame[j] = 1
        didMove.append(didMoveThisFrame)

        # plt.clf()
        # coords = np.array([transform_coord(u["pos"]) for u in playerUnits])
        # plt.scatter(coords[:, 0], coords[:, 1], c=didMoveThisFrame)
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        # plt.pause(0.1)

    return didMove


def extractOrderChanged(rawUnits, playerID, map_size, loader, mirror):
    def transform_coord(coord):
        mn, mx, flipX, flipY = map_size
        normalized_x = (coord["x"] - mn[0]) / (mx[0] - mn[0])
        normalized_y = (coord["y"] - mn[1]) / (mx[1] - mn[1])

        if flipX != mirror:
            normalized_x = 1 - normalized_x
        if flipY != mirror:
            normalized_y = 1 - normalized_y
        return (normalized_x, normalized_y)

    # ["observations"][0]["rawUnits"]
    lastKnownPositions = {}
    didMove = []
    moves = []
    for i in range(len(rawUnits)):
        moves.append(set())

    movementThreshold = 5

    for i, state in enumerate(rawUnits):
        units = state["units"]
        for unit in units:
            if unit["owner"] != playerID:
                continue

            tag = unit["tag"]
            newPos = unitDestination(unit)
            newPos = (newPos["x"], newPos["y"])

            if tag not in lastKnownPositions:
                lastKnownPositions[tag] = newPos
                moves[i].add(tag)
            else:
                oldPos = lastKnownPositions[tag]
                if squaredDistance(oldPos, newPos) > movementThreshold * movementThreshold:
                    # Register movement
                    moves[i - 1].add(tag)
                    lastKnownPositions[tag] = newPos

    for i in range(len(rawUnits)):
        playerUnits = [u for u in rawUnits[i]["units"] if u["owner"] == playerID and isMovableUnit(u, loader)]
        numPlayerUnits = len(playerUnits)
        didMoveThisFrame = torch.zeros(numPlayerUnits, dtype=torch.long)
        for j, u in enumerate(playerUnits):
            if u["tag"] in moves[i]:
                didMoveThisFrame[j] = 1
        didMove.append(didMoveThisFrame)

        # plt.clf()
        # coords = np.array([transform_coord(u["pos"]) for u in playerUnits])
        # plt.scatter(coords[:, 0], coords[:, 1], c=didMoveThisFrame)
        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        # plt.pause(0.1)

    return didMove

def isMovableUnit(u, loader):
    if u["unit_type"] not in loader.unit_lookup.unit_index_map:
        return False

    return loader.unit_lookup.movable_units_mask[loader.unit_lookup.unit_index_map[u["unit_type"]]]


def movementUnitStates(rawUnits, loader, playerID, map_size, minimap_size, mirror):
    def transform_coord(coord):
        mn, mx, flipX, flipY = map_size
        normalized_x = (coord["x"] - mn[0]) / (mx[0] - mn[0])
        normalized_y = (coord["y"] - mn[1]) / (mx[1] - mn[1])

        if flipX != mirror:
            normalized_x = 1 - normalized_x
        if flipY != mirror:
            normalized_y = 1 - normalized_y
        return (normalized_x, normalized_y)

    def unitObservation(unit, result, resultCoords):
        assert unit["is_alive"]
        assert unit["display_type"] == 1
        assert not unit["is_blip"]
        assert result.shape == (2 + 4 * minimap_size,)
        health = (unit["health"] + unit["shield"]) / (unit["health_max"] + unit["shield_max"])
        result[0] = loader.unit_lookup.unit_index_map[unit["unit_type"]]
        coords = transform_coord_minimap(unit["pos"], map_size, minimap_size, mirror)

        target_coord = unitDestination(unit)
        target_coord = transform_coord_minimap(target_coord, map_size, minimap_size, mirror)

        resultCoords[0] = coords[0]
        resultCoords[1] = coords[1]

        # X, Y
        offset = 1
        result[offset + coords[0]] = 1
        offset += minimap_size
        result[offset + coords[1]] = 1
        offset += minimap_size

        # result[offset] = unit["owner"] * 2 - 3  # Convert owner (1|2) to -1 or 1
        # if mirror:
        #     result[9] *= -1
        # assert result[9] == -1 or result[9] == 1
        result[offset] = health
        offset += 1

        result[offset + target_coord[0]] = 1
        offset += minimap_size
        result[offset + target_coord[1]] = 1
        offset += minimap_size

        return result

    valid_units = []
    for unit in rawUnits["units"]:
        if unit["owner"] == playerID and isMovableUnit(unit, loader):
            valid_units.append(unit)

    # plt.clf()
    # plt.scatter(x=[transform_coord(u["pos"])[0] for u in valid_units[0]], y=[transform_coord(u["pos"])[1] for u in valid_units[0]], c="#FF0000")
    # plt.scatter(x=[transform_coord(u["pos"])[0] for u in valid_units[1]], y=[transform_coord(u["pos"])[1] for u in valid_units[1]], c="#0000FF")

    # plt.imshow(total_minimap_health, interpolation='nearest', cmap=plt.cm.inferno)
    # plt.colorbar()
    # plt.clf()
    # colors = ["#FF0000" if u["owner"] == 1 else "#0000FF" for u in combined_units]
    # xs = list(range(len(combined_units)))
    # plt.bar(xs, [u["health"] + u["shield"] for u in combined_units], color=colors)
    # plt.xlim([0, num_input_units])
    # # plt.show()
    # plt.pause(0.001)

    allUnits = np.zeros((len(valid_units), 2 + 4 * minimap_size), dtype=np.float32)
    unitCoordinates = np.zeros((len(valid_units), 2), dtype=np.long)

    for i, unit in enumerate(valid_units):
        unitObservation(unit, result=allUnits[i], resultCoords=unitCoordinates[i])

    return torch.tensor(allUnits), torch.tensor(unitCoordinates)
