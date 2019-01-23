from build_order_loader import BuildOrderLoader, to_one_hot
from collections import namedtuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

Trace = namedtuple('Trace', ['states', 'winner', 'replay_path', 'minimap_states'])

swap_winner_replays = [
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/231ccb5bdf905a4b7246a6800876fd03ca1eb7b0407ac9753c754e6675a63e44.SC2Replay",  # Other player rage-quits
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/2d44e2f72f7147b3c9068a7417f9676e234dc250947405cf91017005319f99c6.SC2Replay",  # Wrong player seems to quit for some reason

    # Double check
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/16bcf92049063d9abb73f017519cd883aa66092e2d81656ab254bc024e815648.SC2Replay",
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/061b34556dfcc36d7999e9a99faa4056a96057b1b50aa79afa90c433e25c5113.SC2Replay",  # Wrong player seems to quit for some reason
]

blacklisted_replays = [
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/144e15f5c4f42769c27a897680bbf0ebeebc8042250bdcd7a8ab7a9711e5e89c.SC2Replay",  # Weird wall of planetary fortresses
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/0cf5d8f5cba9023cf4cb09f56711f51f50d2126bb781bfedcbfc2e3a1e45f7b8.SC2Replay",  # SCV rush that failed... but the opponent still seems to give up?
]

whitelisted_replays = [
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/046e6ab6b6d9cce5985493bff4f5fac7bc230395107f6e1dcceacd99c2248563.SC2Replay",  # Sneaky battlecruiser attack
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/13f4e78a4a222a11954f16b684e2c72eb6d317e9615daeccc3d6af16d0e3d769.SC2Replay",  # another sneaky battlecruiser attack
    "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays/2874a258f9360311715aa66394060e165d318a6889af88cf47f015d39e095150.SC2Replay",  # Strange base-trade
]


def armyStrength(loader, state):
    count = 0
    for u in state["units"]:
        try:
            count += (loader.unit_lookup.military_units_mask[loader.unit_lookup.unit_index_map[u["type"]]]*0.9 + 0.1) * u["totalCount"]
        except Exception:
            pass

    return count


def getInputTensorSize(loader: BuildOrderLoader):
    # return loader.get_tensor_size(loader.unit_lookup.all_unit_indices, False) + 2
    return 7 + 2 + loader.unit_lookup.num_upgrades


def find_map_size(session):
    def start_unit(units):
        for u in units:
            if u["health"] > 100:
                return u

        assert False

    start_unit1 = start_unit(session["observations"][0]["rawUnits"][0]["units"])
    start_unit2 = start_unit(session["observations"][1]["rawUnits"][0]["units"])
    mn_x = min(start_unit1["pos"]["x"], start_unit2["pos"]["x"])
    mn_y = min(start_unit1["pos"]["y"], start_unit2["pos"]["y"])
    mx_x = max(start_unit1["pos"]["x"], start_unit2["pos"]["x"])
    mx_y = max(start_unit1["pos"]["y"], start_unit2["pos"]["y"])

    s = max(mx_x/2, mx_y/2)
    mn_x = 0
    mn_y = 0
    mx_x = 2*s
    mx_y = 2*s

    # print(((mn_x, mn_y), (mx_x, mx_y)), session["replayInfo"]["replay_path"])
    return ((mn_x, mn_y), (mx_x, mx_y))


# class SessionContext:
#     def __init__(session):
#         self.map_size = find_map_size(session)


# def loadState(context, observation1, observations2):
#     player1obs = playerObservationTensor(loader, observation1["selfStates"]) for x in 
#     player2obs = playerObservationTensor(loader, observation2["selfStates"]) for x in 
#     player1obs2 = playerObservationTensor2(loader, s, r, 1, map_size) for (s,r) in zip(observation1["selfStates"], observation1["rawUnits"])
#     player2obs2 = playerObservationTensor2(loader, s, r, 2, map_size) for (s,r) in zip(observation2["selfStates"], observation2["rawUnits"])

def loadSession(session, loader: BuildOrderLoader, memory, statistics):
    if session["mmrs"][0] < 3000 and session["mmrs"][1] < 3000:
        print("Skipping due to low MMR")
        return

    if session["winner"] == -1:
        print("Unknown winner, skipping replay")
        return

    gameDurationSeconds = session["replayInfo"]["duration_gameloops"]/22.4

    # Skip very short games (very likely a player quitting due to some other reason)
    if gameDurationSeconds < 60:
        print("Skipping short game")
        return

    replay_path = session["replayInfo"]["replay_path"]

    if replay_path in blacklisted_replays:
        print("Skipping blacklisted replay")
        return

    map_size = find_map_size(session)
    player1obs = [playerObservationTensor(loader, x) for x in session["observations"][0]["selfStates"]]
    player2obs = [playerObservationTensor(loader, x) for x in session["observations"][1]["selfStates"]]
    player1obs2 = [playerObservationTensor2(loader, s, r, 1, map_size) for (s,r) in zip(session["observations"][0]["selfStates"], session["observations"][0]["rawUnits"])]
    player2obs2 = [playerObservationTensor2(loader, s, r, 2, map_size) for (s,r) in zip(session["observations"][1]["selfStates"], session["observations"][1]["rawUnits"])]

    # player1obs = [np.concatenate([a,b]) for (a,b) in zip(player1obs, player1obs2)]
    # player2obs = [np.concatenate([a,b]) for (a,b) in zip(player2obs, player2obs2)]
    player1obs = player1obs2
    player2obs = player2obs2

    states = [np.stack([a, b]) for ((a,_),(b,_)) in zip(player1obs, player2obs)]
    mirrored_states = [np.stack([b, a]) for ((a,_),(b,_)) in zip(player1obs, player2obs)]
    minimap_states = [np.stack([a, b]) for ((_,a),(_,b)) in zip(player1obs, player2obs)]
    mirrored_minimap_states = [np.stack([b, a]) for ((_,a),(_,b)) in zip(player1obs, player2obs)]

    # Convert from 1-indexed to 0-indexed
    winner = session["winner"] - 1
    assert winner == 0 or winner == 1

    if replay_path in swap_winner_replays:
        print("Swapping winner for replay")
        winner = 1 - winner

    p1s = armyStrength(loader, session["observations"][winner]["selfStates"][-1])
    p2s = armyStrength(loader, session["observations"][1 - winner]["selfStates"][-1])

    if p1s > p2s:
        pass
    else:
        print("Not consistent ", p1s, p2s)
        if p1s < p2s/3.0:
            if replay_path in whitelisted_replays:
                print("Very not consistent, however this replay is whitelisted")
            else:
                print(f"Very not consistent ({p1s} < {p2s})", replay_path)
                print("Skipping")
                return

    trace = Trace(states=states, winner=winner, replay_path=replay_path, minimap_states=minimap_states)
    mirror = Trace(states=mirrored_states, winner=1 - winner, replay_path=replay_path, minimap_states=mirrored_minimap_states)
    memory.push(trace)
    memory.push(mirror)


def playerObservationTensor(loader: BuildOrderLoader, state):
    tensor = loader.createState(state, unitIndices=loader.unit_lookup.all_unit_indices)
    return tensor[0]


def playerObservationTensor2(loader: BuildOrderLoader, state, raw_units, playerID, map_size):
    units = raw_units["units"]
    total_army_health = 0
    total_army_health_fraction = 0.000001
    total_army_health_fraction_weight = 0.000001

    minimap_size = 5

    minimap_health = np.zeros((minimap_size, minimap_size, loader.num_units))
    minimap_count = np.zeros((minimap_size, minimap_size, loader.num_units))
    flying_health = np.zeros((minimap_size, minimap_size, loader.num_units))

    def isArmy(unit_type):
        if unit_type in loader.unit_lookup.unit_index_map:
            return loader.unit_lookup.military_units_mask[loader.unit_lookup.unit_index_map[unit_type]] > 0
        else:
            return False

    def transform_coord(coord):
        normalized_x = (coord["x"] - map_size[0][0])/(map_size[1][0] - map_size[0][0])
        normalized_y = (coord["y"] - map_size[0][1])/(map_size[1][1] - map_size[0][1])
        normalized_x *= minimap_size
        normalized_y *= minimap_size
        return (min(minimap_size-1, max(0, round(normalized_x))), min(minimap_size-1, max(0, round(normalized_y))))

    for unit in units:
        if not unit["is_alive"]:
            continue

        if unit["owner"] == playerID:
            unit_type = unit["unit_type"]
            if unit_type in loader.unit_lookup.unit_index_map:
                coord = transform_coord(unit["pos"])
                index = loader.unit_lookup.unit_index_map[unit_type]
                minimap_count[coord[0], coord[1], index] += 1
                if unit["is_flying"]:
                    flying_health[coord[0], coord[1], index] += (unit["health"] + unit["shield"])/(unit["health_max"] + unit["shield_max"])
                else:
                    minimap_health[coord[0], coord[1], index] += (unit["health"] + unit["shield"])/(unit["health_max"] + unit["shield_max"])

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

    return np.concatenate([metaTensor, upgradeTensor, np.array([total_army_health / 1000, total_army_health_fraction])]), minimap_health
