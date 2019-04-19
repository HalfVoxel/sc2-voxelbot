#include "rl_planning_env.h"
#include "tensor.h"
#include "../mcts/mcts_debugger.h"

using namespace std;
using namespace sc2;

const float TimestepDuration = 3.0f;

MCTSDebugger* simulatorVisualizer;

pybind11::array_t<float> oneHotEncode(vector<pair<int, float>> items) {
    int totalLength = 0;
    for (auto p : items) totalLength += p.first;
    pybind11::array_t<float> res(totalLength);
    npZero(res);
    int offset = 0;
    for (auto p : items) {
        float v = min(p.first - 1.0f, max(0.0f, p.second));
        // TOOD: Smooth one hot
        res.mutable_at(offset + round(v)) = 1;
        offset += p.first;
    }
    return res;
}

vector<pybind11::array_t<float>> EnvObserver::observe(const SimulatorMCTSState& state, int playerID) const {
    assert(playerID == 1 || playerID == 2);

    int minimapSize = 16;
    auto minimapOurBuildings = Tensor(minimapSize, minimapSize);
    auto minimapOurArmy = Tensor(minimapSize, minimapSize);
    auto minimapEnemy = Tensor(minimapSize, minimapSize);
    auto minimapEnemyBuildings = Tensor(minimapSize, minimapSize);
    auto minimapOurFlying = Tensor(minimapSize, minimapSize);

    const SimulatorState& simState = state.state;

    for (auto& g : simState.groups) {
        if (g.owner == playerID) {
            for (auto& u : g.units) {
                auto pos = coordinates.transformCell(g.pos, minimapSize);
                if (isArmy(u.combat.type)) {
                    minimapOurArmy(pos) += 1;
                }
                if (isStructure(u.combat.type)) {
                    minimapOurBuildings(pos) += 1;
                }
                if (u.combat.is_flying) {
                    minimapOurFlying(pos) += 1;
                }
            }
        } else {
            for (auto& u : g.units) {
                auto pos = coordinates.transformCell(g.pos, minimapSize);
                if (isArmy(u.combat.type)) {
                    minimapEnemy(pos) += 1;
                }
                if (isStructure(u.combat.type)) {
                    minimapEnemyBuildings(pos) += 1;
                }
            }
        }
    }

    auto buildState = simState.states[playerID - 1];

    CombatState cs;
    for (auto& g : simState.groups) {
        for (auto& u : g.units) {
            // TODO: Or static defense?
            if (isArmy(u.combat.type)) {
                cs.units.push_back(u.combat);
            }
        }
    }

    CombatSettings combatSettings;
    int defenderPlayer = 3 - playerID;
    CombatResult combatResult = state.state.simulator->combatPredictor->predict_engage(cs, combatSettings, nullptr, defenderPlayer);

    auto meta = oneHotEncode({
        { 5, buildState->resources.minerals/100 },
        { 5, buildState->resources.vespene/100 },
        { 5, buildState->miningSpeed().mineralsPerSecond/20 },
        { 5, buildState->miningSpeed().vespenePerSecond/20 },
        { 5, buildState->baseInfos.size() },
    });

    auto meta2 = Tensor(6, 1);
    meta2[0] = combatResult.state.owner_with_best_outcome() == playerID ? 1 : 0;
    meta2[1] = combatResult.time;
    meta2[2] = minimapOurArmy.sum();
    meta2[3] = minimapOurBuildings.sum();
    meta2[4] = minimapEnemy.sum();
    meta2[5] = minimapEnemyBuildings.sum();

    auto unitTypesOur = Tensor(unitMappings.size(), 1);
    auto unitTypesEnemy = Tensor(unitMappings.size(), 1);
    for (auto& g : simState.groups) {
        for (auto& u : g.units) {
            int idx = unitMappings.getIndexMaybe(u.combat.type);
            if (idx != -1) {
                if (g.owner == playerID) {
                    unitTypesOur(idx) += 1;
                } else {
                    unitTypesEnemy(idx) += 1;
                }
            }
        }
    }

    auto unitTypesOurOneHot = Tensor(unitMappings.size(), 5);
    auto unitTypesEnemyOneHot = Tensor(unitMappings.size(), 5);
    for (int i = 0; i < unitMappings.size(); i++) {
        if (unitTypesOur[i] == 0) {
            unitTypesOurOneHot(i, 0) = 1;
        }
        if (unitTypesOur[i] >= 1) {
            unitTypesOurOneHot(i, 1) = 1;
        }
        if (unitTypesOur[i] >= 3) {
            unitTypesOurOneHot(i, 2) = 1;
        }
        if (unitTypesOur[i] >= 9) {
            unitTypesOurOneHot(i, 3) = 1;
        }
        if (unitTypesOur[i] >= 27) {
            unitTypesOurOneHot(i, 4) = 1;
        }

        if (unitTypesEnemy[i] == 0) {
            unitTypesEnemyOneHot(i, 0) = 1;
        }
        if (unitTypesEnemy[i] >= 1) {
            unitTypesEnemyOneHot(i, 1) = 1;
        }
        if (unitTypesEnemy[i] >= 3) {
            unitTypesEnemyOneHot(i, 2) = 1;
        }
        if (unitTypesEnemy[i] >= 9) {
            unitTypesEnemyOneHot(i, 3) = 1;
        }
        if (unitTypesEnemy[i] >= 27) {
            unitTypesEnemyOneHot(i, 4) = 1;
        }
    }

    return { move(minimapOurBuildings.weights), move(minimapOurArmy.weights),  move(minimapEnemy.weights), move(minimapEnemyBuildings.weights), move(minimapOurFlying.weights), move(meta), move(meta2.weights), move(unitTypesOurOneHot.weights), move(unitTypesEnemyOneHot.weights) };
}

pair<float, bool> rewardFromStateDelta(const SimulatorState& state1, const SimulatorState& state2, int playerID) {
    map<Tag, const SimulatorUnit*> tagMapping;
    for (auto& g : state2.groups) {
        for (auto& u : g.units) {
            tagMapping[u.tag] = &u;
        }
    }

    float reward = 0;

    int allyUnitCount = 0;
    int enemyUnitCount = 0;

    for (auto& g : state1.groups) {
        for (auto& u : g.units) {
            if (g.owner == playerID) allyUnitCount++;
            else enemyUnitCount++;

            auto* previousUnit = tagMapping[u.tag];

            if (g.owner == playerID) {
                // float newHealth = previousUnit == nullptr ? 0 : previousUnit->combat.health + previousUnit->combat.shield;
                if (previousUnit == nullptr) {
                    // Lost unit! Oh no!
                    auto& data = getUnitData(u.combat.type);
                    reward -= data.mineral_cost + data.vespene_cost;
                }
            } else {
                if (previousUnit == nullptr) {
                    // Killed unit! Yay!
                    auto& data = getUnitData(u.combat.type);
                    reward += data.mineral_cost + data.vespene_cost;
                }
            }
        }
    }

    // Yay, enemy is dead!
    if (enemyUnitCount == 0) {
        reward += 10000;
    }

    if (allyUnitCount == 0) {
        reward -= 10000;
    }

    bool done = enemyUnitCount == 0 || allyUnitCount == 0;

    // Prevent very long games
    if (state2.time() > 60*40) {
        reward -= 5000;
        done = true;
    }

    return { reward, done };
}

std::pair<float, bool> RLPlanningEnv::step(int actionIndex) {
    MCTSAction action = (MCTSAction)actionIndex;
    assert(action >= (MCTSAction)0 && action < MCTSAction::Count);

    state.player = trainingPlayer - 1;
    if (!state.executeAction(action)) {
        // NOOP action? allow? add to observation which ones are noops?
    }

    // TODO: Some strategy for other player?
    state.player = (3 - trainingPlayer) - 1;
    state.executeAction(MCTSAction::ArmyAttackClosestEnemy);

    auto originalState = state.state;
    state.state.simulate(state.state.time() + TimestepDuration);

    pair<float, bool> rewardAndDone = rewardFromStateDelta(originalState, state.state, trainingPlayer);

    return rewardAndDone;
}

vector<pybind11::array_t<float>> RLPlanningEnv::observe() {
    return observer.observe(state, trainingPlayer);
}

string RLPlanningEnv::actionName(int action) {
    return MCTSActionName((MCTSAction)action);
}

pair<vector<vector<double>>, float> RLPlanningEnv::visualizationInfo() {
    vector<vector<double>> units;
    for (auto& group : state.state.groups) {
        for (auto& u : group.units) {
            units.push_back(vector<double> { group.pos.x, group.pos.y, (double)u.combat.owner, (double)u.tag, (double)u.combat.type, u.combat.health + u.combat.shield, u.combat.health_max + u.combat.shield_max });
        }
    }

    return { units, state.state.time() };
}

void RLPlanningEnv::print() {
    simulatorVisualizer->visualize(state.state);
}

RLEnvManager::RLEnvManager(pybind11::object simulatorVisualizerModule, pybind11::object replayLoadFn, vector<string> binaryReplayFilePaths) :
    observer(getAvailableUnitsForRace(Race::Protoss), CoordinateRemapper(Point2D(0,0), Point2D(100, 100), false, false))
{
    simulatorVisualizer = new MCTSDebugger(simulatorVisualizerModule);
    initMappings();
    combatPredictor.init();

    for (int i = 0; i < 79; i++) {
        cout << "Loading replay " << i << " " << binaryReplayFilePaths[i] << endl;
        auto session = loadReplayBinary(replayLoadFn(binaryReplayFilePaths[i]).cast<string>());
        if (!filterSession(session)) continue;
        statePool.push_back(loadPlanningEnvSession(session));
    }
}

SimulatorState createStartingState(shared_ptr<SimulatorContext> simulator);

RLPlanningEnv RLEnvManager::getEnv() {
    int playerID = 1;
    auto simulator = make_shared<SimulatorContext>(&combatPredictor, vector<Point2D>{ Point2D(20,20), Point2D(168 - 20, 168 - 20) });
    SimulatorState simulatorState = statePool[rand() % statePool.size()]; //createStartingState(simulator);
    simulatorState.simulator = simulator;
    SimulatorMCTSState mctsState(simulatorState, playerID - 1);
    RLPlanningEnv env(playerID, mctsState, observer);
    return env;
}

SimulatorState createStartingState(shared_ptr<SimulatorContext> simulator) {
    BuildOrder bo1 = { UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_PYLON,
    };
    BuildOrder bo2 = { UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_ZEALOT,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
        UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_VOIDRAY, UNIT_TYPEID::PROTOSS_PYLON,
    };
    SimulatorState state(simulator, {
        // simulator->cache.copyState(BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PYLON, 6 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 0 }, { UNIT_TYPEID::PROTOSS_STALKER, 6 }, { UNIT_TYPEID::PROTOSS_PHOENIX, 9 }, { UNIT_TYPEID::PROTOSS_IMMORTAL, 12 }})),
        // simulator->cache.copyState(BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 2 }, { UNIT_TYPEID::PROTOSS_PYLON, 6 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 }, { UNIT_TYPEID::PROTOSS_CARRIER, 2 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 1+2 }, { UNIT_TYPEID::PROTOSS_COLOSSUS, 2 }, { UNIT_TYPEID::PROTOSS_VOIDRAY, 2 }, { UNIT_TYPEID::PROTOSS_PHOTONCANNON, 6 }, { UNIT_TYPEID::PROTOSS_STARGATE, 1 } })),
        simulator->cache.copyState(BuildState()),
        simulator->cache.copyState(BuildState())
    },
    {
        BuildOrderState(make_shared<BuildOrder>(bo1)),
        BuildOrderState(make_shared<BuildOrder>(bo2)),
    });

    // state.states[1]->resources.minerals = 10000;
    // state.states[1]->resources.vespene = 10000;

    state.groups.push_back(SimulatorUnitGroup(Point2D(10, 10), {
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PHOENIX)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(20, 10), {
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PROBE)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(20, 10), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PROBE)),
    }));

    if ((rand() % 2) == 0) {
        state.groups.push_back(SimulatorUnitGroup(Point2D(100, 100), {
            SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT)),
            SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER)),
            SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER)),
        }));
    }

    if ((rand() % 2) == 0) {
        state.groups.push_back(SimulatorUnitGroup(Point2D(50, 10), {
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        }));
    }

    state.groups.push_back(SimulatorUnitGroup(Point2D(80, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(5, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_VOIDRAY)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(100, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_NEXUS)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_STARGATE)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(50, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_NEXUS)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(10, 10), {
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_NEXUS)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_PYLON)),
    }));

    if ((rand() % 2) == 0) {
        state.groups.push_back(SimulatorUnitGroup(Point2D(100, 40), {
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
            SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        }));
    }

    // Replace with correct build states
    vector<BuildState> states(2);
    for (auto& g : state.groups) {
        for (auto& u : g.units) {
            states[g.owner - 1].addUnits(u.combat.type, 1);
        }
    }
    state.states = { simulator->cache.copyState(states[0]), simulator->cache.copyState(states[1]) };

    return state;
}
