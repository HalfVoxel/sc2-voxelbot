#include "game_state_loader.h"
#include <cereal/cereal.hpp>
#include "../utilities/cereal_json.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include "../unit_lists.h"
#include <fstream>

using namespace std;
using namespace sc2;

Point2D CoordinateRemapper::transform (Point2D coord) const {
    // Normalize
    float nx = (coord.x - mn.x) / (mx.x - mn.x);
    float ny = (coord.y - mn.y) / (mx.y - mn.y);

    if (flipX) nx = 1 - nx;
    if (flipY) ny = 1 - ny;

    return Point2D(nx, ny);
}

Point2DI CoordinateRemapper::transformCell(Point2D coord, int scale) const {
    auto p = transform(coord);
    int x = (int)round(p.x * scale);
    int y = (int)round(p.y * scale);
    x = min(scale - 1, max(0, x));
    y = min(scale - 1, max(0, y));
    return Point2DI(x, y);
}

Point2D CoordinateRemapper::inverseTransform(Point2D coord) const {
    float nx = coord.x;
    float ny = coord.y;

    if (flipX) nx = 1 - nx;
    if (flipY) ny = 1 - ny;

    float x = nx * (mx.x - mn.x) + mn.x;
    float y = ny * (mx.y - mn.y) + mn.y;

    return Point2D(x, y);
}

// ReplaySession loadReplayBinary(string filepath) {
//     ReplaySession session;
//     {
//         ifstream json3(filepath);
//         cereal::BinaryInputArchive archive3(json3);
//         session.serialize(archive3);
//         json3.close();
//     }
//     cout << "Loaded replay" << endl;
//     return session;
// }
ReplaySession loadReplayBinary(std::string f) {
    // Stopwatch w;
    ReplaySession session;
    {
        stringstream json3(f);
        cereal::BinaryInputArchive archive3(json3);
        session.serialize(archive3);
    }
    // w.stop();
    // cout << w.millis() << endl;
    return session;
}

bool filterSession(const ReplaySession& session) {
    if (session.mmrs[0] < 3000 && session.mmrs[1] < 3000) {
        cout << "Skipping due to low MMR" << endl;
        return false;
    }

    if (session.winner == -1) {
        cout << "Unknown winner, skipping replay" << endl;
        return false;
    }

    if (session.winner == 0) {
        cout << "Tie, skipping replay" << endl;
        return false;
    }

    if (session.winner != 1 && session.winner != 2) {
        cout << "Unexpected winner " << session.winner << ", skipping" << endl;
        return false;
    }

    if (getUnitData(session.observations[0].selfStates[0].units[0].type).race != Race::Protoss || getUnitData(session.observations[1].selfStates[0].units[0].type).race != Race::Protoss) {
        cout << "Not PvP, skipping. Was " << RaceToString(getUnitData(session.observations[0].selfStates[0].units[0].type).race) << "v" << RaceToString(getUnitData(session.observations[1].selfStates[0].units[0].type).race) << endl;
        return false;
    }

    float gameDurationSeconds = session.replayInfo.duration_gameloops / 22.4f;

    if (gameDurationSeconds < 60) {
        cout << "Skipping short game" << endl;
        return false;
    }

    if (gameDurationSeconds > 40 * 60) {
        cout << "Skipping long game" << endl;
        return false;
    }

    string map_name = session.gameInfo.map_name;
    if (map_name == "Stasis LE") {
        cout << "Skipping blacklisted map " << map_name << endl;
        return false;
    }

    return true;
}

struct ExtractedBuildOrder {
    BuildOrder buildOrder;
    vector<Point2D> spawnPositions;
};

ExtractedBuildOrder extractBuildOrder(const ReplaySession& session, int playerID, int startingTimestep) {
    auto& obs = session.observations[playerID - 1];

    ExtractedBuildOrder bo;
    set<Tag> seenTags;
    AvailableUnitTypes unitTypes = getAvailableUnitsForRace(Race::Protoss);

    for (int t = startingTimestep; t < (int)obs.rawUnits.size(); t++) {
        auto& units = obs.rawUnits[t].units;
        for (auto& unit : units) {
            if (unit.owner == playerID) {
                if (!unitTypes.contains(unit.unit_type)) continue;

                // Can't really handle this one yet
                if (unit.unit_type == UNIT_TYPEID::PROTOSS_MOTHERSHIP) continue;

                if (!seenTags.count(unit.tag)) {
                    seenTags.insert(unit.tag);

                    if (t != startingTimestep) {
                        bo.buildOrder.items.push_back(BuildOrderItem(unit.unit_type, false));
                        bo.spawnPositions.push_back(unit.pos);
                    }
                }
            }
        }
    }

    return bo;
}

SimulatorState extractSimulatorState (const ReplaySession& session, int timestep) {
    array<shared_ptr<BuildState>, 2> buildStates = {{ make_shared<BuildState>(), make_shared<BuildState>() }};
    auto bo1 = extractBuildOrder(session, 1, timestep);
    auto bo2 = extractBuildOrder(session, 2, timestep);
    SimulatorState state(nullptr, { buildStates[0], buildStates[1] }, { BuildOrderState(make_shared<BuildOrder>(bo1.buildOrder)), BuildOrderState(make_shared<BuildOrder>(bo2.buildOrder)) });

    for (int playerID = 1; playerID <= 2; playerID++) {
        auto& obs = session.observations[playerID - 1];
        for (auto& unit : obs.rawUnits[timestep].units) {
            if (unit.owner == playerID) {
                state.addUnit(&unit);
                buildStates[unit.owner - 1]->addUnits(unit.unit_type, 1);
            }
        }

        BuildState& buildState = *buildStates[playerID - 1];
        buildState.resources.minerals = obs.selfStates[timestep].minerals;
        buildState.resources.vespene = obs.selfStates[timestep].vespene;
        buildState.race = obs.selfStates[timestep].race;

        // TODO:
        // obs.selfStates[timestep].upgrades;
    }

    return state;
}

SimulatorState loadPlanningEnvSession(const ReplaySession& session) {
    int timesteps = (int)session.observations[0].rawUnits.size();

    int timestep = rand() % (timesteps/2);
    SimulatorState state = extractSimulatorState(session, timestep);

    return state;
}


#if false


vector<float> concat(const vector<float>& a, const vector<float>& b) {
    vector<float> res;
    for (auto v : a) res.push_back(v);
    for (auto v : b) res.push_back(v);
    return res;
}

vector<float> concat(const vector<float>& a, const vector<float>& b, const vector<float>& c) {
    vector<float> res;
    for (auto v : a) res.push_back(v);
    for (auto v : b) res.push_back(v);
    for (auto v : c) res.push_back(v);
    return res;
}

vector<float> globalObservationTensor(const SerializedState& state, const RawState& rawUnits, int playerID, CoordinateRemapper map_size, int minimapSize) {
    auto units = rawUnits.units;
    float total_army_health = 0;
    float total_army_health_fraction = 0.000001;
    float total_army_health_fraction_weight = 0.000001;
    int total_units = 0;

    for (auto& unit : units) {
        if (!unit.is_alive) continue;

        if unit.owner == playerID:
            auto unit_type = unit.unit_type;
            // if unitMappings.contains(unit_type):
            //     coord = mapSize.transformCell(unit.pos, minimapSize)
            //     index = unitMappings.toIndex(unit_type)
            total_units += 1;

            if isArmy(unit_type):
                total_army_health += unit.health + unit.shield;
                total_army_health_fraction += unit.health + unit.shield;
                total_army_health_fraction_weight += (unit.health_max + unit.shield_max);

    total_army_health_fraction /= total_army_health_fraction_weight

    vector<float> metaTensor {
        state.minerals / 100.0f,
        state.vespene / 100.0f,
        state.mineralsPerSecond / 10.0f,
        state.vespenePerSecond / 10.0f,
        state.highYieldMineralSlots / 10.0f,
        state.lowYieldMineralSlots / 10.0f,
        total_units / 50.0f,
    };

    vector<float> upgradeTensor (unitMappings.numUpgrades);
    for (UPGRADE_ID u : state.upgrades) {
        if (unitMappings.contains(u)) {
            upgradeTensor[unitMappings.toIndex(u)] = 1;
        }
    }

    armyTensor = vector<float> { total_army_health / 1000.0f, total_army_health_fraction };
    return concat(metaTensor, upgradeTensor, armyTensor);

vector<InfluenceMap> minimapLayers(const RawState& rawUnits, int playerID, CoordinateRemapper mapSize, int minimapSize) {
    auto units = rawUnits.units;

    minimapHealth = InfluenceMap(minimap_size, minimap_size);
    minimapCount = InfluenceMap(minimap_size, minimap_size);
    flyingHealth = InfluenceMap(minimap_size, minimap_size);
    minimapHealthBuildings = InfluenceMap(minimap_size, minimap_size);
    minimapHealthWorkers = InfluenceMap(minimap_size, minimap_size);

    for (auto& unit : units) {
        if (!unit.is_alive) continue;

        if (unit.owner == playerID) {
            UNIT_TYPEID unit_type = unit.unit_type;
            if unitMappings.contains(unit_type) {
                auto coord = mapSize.transformCell(unit.pos, minimapSize);
                int index = unitMappings.toIndex(unitType);
                minimapCount[coord] += 1;
                float maxHealth = max(unit.health_max + unit.shield_max, 1);
                if (isArmy(unit_type)) {
                    if (unit.is_flying) {
                        flyingHealth[coord] += (unit.health + unit.shield);
                    } else {
                        minimapHealth[coord] += (unit.health + unit.shield);
                    }
                }

                if (!isMovableUnit(unit, loader)) {
                    minimapHealthBuildings[coord] += (unit.health + unit.shield) / max_health;
                }

                if (isBasicWorker(unit_type)) {
                    minimapHealthWorkers[coord] += (unit.health + unit.shield) / max_health;
                }
            }
        }
    }

    minimap = vector<InfluenceMap> {
        move(minimapHealth),
        move(minimapCount),
        move(flyingHealth),
        move(minimapHealthBuildings),
        move(minimapHealthWorkers),
    };
    return minimap

struct MovementTargetTrace {
    vector<vector<float>> states;
    vector<vector<InfluenceMap>> minimap_states;
    string replay_path;
    vector<vector<Point2DI>> target_positions;
    vector<vector<int>> unit_type_counts;
    vector<float> fraction_similar_orders;
    vector<float> attack_order;
    string data_path;
    int playerID;
    vector<InfluenceMap> pathfinding_minimap;
};

GameStateLoader::loadMovementTarget (int playerID, set<Tag>* unit_tag_mask) {
    res = loadSessionMovementTarget2(playerID, 'random');

    int playerIndex = playerID - 1;
    int opponentPlayerID = 3 - playerID;

    auto mapSize = findMapSize(session, playerID);
    auto& selfStates = session.observations.selfStates;
    auto& rawUnits = session.observations.rawUnits;
    int minimapSize = 14;

    int lookaheadTime = unit_tag_mask == nullptr ? 4 : 0;
    int maxTime = rawUnits.size() - lookaheadTime;

    if (maxTime <= 0) {
        cout << "Skipping game with too few sampled";
        return;
    }

    vector<vector<float>> globalObservations;
    vector<vector<InfluenceMap>> minimaps;

    vector<vector<Point2DI>> targetPositions;
    vector<float> fractionSimilarOrders;
    vector<float> attackOrder;
    vector<vector<int>> unitTypeCounts;
    vector<vector<InfluenceMap>> movementMinimap;

    for (int t = 0; t < maxTime; t++) {
        globalObservations.push_back(move(globalObservationTensor(selfStates[t], rawUnits[t], playerID, mapSize, minimapSize)));
        auto minimap = minimapLayers(rawUnits[t], playerID, mapSize, minimapSize);
        for (auto l : minimapLayers(rawUnits[t], opponentPlayerID, mapSize, minimapSize)) {
            minimap.push_back(move(l));
        }
        minimaps.push_back(move(minimap));

        vector<InfluenceMap> movementMinimapLayer;
        vector<int> unitTypeCountsLayer;
        vector<Point2DI> targetPositionsLayer;
        float fractionSimilarOrdersLayer;
        float attackOrderLayer;
        tie(movementMinimapLayer, unitTypeCountsLayer, targetPositionsLayer, fractionSimilarOrdersLayer, attackOrderLayer) = sampleMovementGroupOrder(rawUnits[t], rawUnits[t+1], playerID, mapSize, minimapSize);

        movementMinimap.push_back(move(movementMinimapLayer));
        unitTypeCounts.push_back(move(unitTypeCountsLayer));
        targetPositions.push_back(move(targetPositionsLayer));
        fractionSimilarOrders.push_back(fractionSimilarOrdersLayer);
        attackOrder.push_back(attackOrderLayer);
    }
}

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
        player1movementMinimap, unit_type_counts, target_positions, fractionSimilarOrders, attack_order = sampleMovementGroupOrder(rawUnits, playerID, map_size, loader, minimap_size, mirror, lookaheadTime)
    else:
        target_positions = None
        assert len(rawUnits) == 1
        units = [u for u in rawUnits[0]["units"] if u["tag"] in unit_tag_mask]
        unitsComplement = [u for u in rawUnits[0]["units"] if u["owner"] == playerID and isMovableUnit(u, loader) and u["tag"] not in unit_tag_mask]
        player1movementMinimap, unit_type_counts, fractionSimilarOrders, attack_order = movementGroupFeatures(units, None, unitsComplement, rawUnits[0]["units"], None, loader, map_size, minimap_size, mirror)
        player1movementMinimap = torch.tensor(player1movementMinimap).unsqueeze(0)
        unit_type_counts = torch.tensor(unit_type_counts).unsqueeze(0)
        fractionSimilarOrders = torch.tensor(fractionSimilarOrders, dtype=torch.float32).unsqueeze(0)

    player1minimap = torch.cat([player1minimap, player1movementMinimap], dim=1)

    return MovementTargetTrace(
        states=torch.stack([x[0] for x in player1obs2]),
        minimap_states=player1minimap,
        replay_path=replay_path,
        target_positions=target_positions,
        unit_type_counts=unit_type_counts,
        fraction_similar_orders=fractionSimilarOrders,
        attack_order=attack_order,
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
#endif