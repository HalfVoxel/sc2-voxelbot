#include "../ml/simulator.h"
#include "../utilities/predicates.h"
#include "../utilities/mappings.h"
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <random>
#include <ctime>
#include <sstream>
#include <iostream>
#include "../mcts/mcts.h"

using namespace std;
using namespace sc2;

function<bool(const SimulatorUnitGroup&)> idleGroup = [](const SimulatorUnitGroup& group) { return group.order.type == SimulatorOrderType::None; };
function<bool(const SimulatorUnitGroup&)> structureGroup = [](const SimulatorUnitGroup& group) { return isStructure(group.units[0].combat.type); };
function<bool(const SimulatorUnit&)> armyUnit = [](const SimulatorUnit& unit) { return isArmy(unit.combat.type); };
function<bool(const SimulatorUnit&)> notArmyUnit = [](const SimulatorUnit& unit) { return !isArmy(unit.combat.type); };
function<bool(const SimulatorUnit&)> structureUnit = [](const SimulatorUnit& unit) { return isStructure(unit.combat.type); };

pybind11::object visualize_fn;

enum class MCTSAction {
    ArmyAttackClosestEnemy,
    IdleArmyAttackClosestEnemy,
    IdleNonArmyAttackClosestEnemy,
    NonArmyAttackClosestEnemy,
    ArmyConsolidate,
    IdleArmyConsolidate,
    ArmyMoveC1,
    ArmyMoveC2,
    None,
    ArmyMoveBase,
    NonArmyMoveBase,
    ArmyAttackBase,
    ArmySuicide,
};

Point2D averagePos(vector<SimulatorUnitGroup*> groups) {
    Point2D p;
    for (auto* g : groups) {
        p += g->pos;
    }
    return p / groups.size();
}

SimulatorUnitGroup* closestGroup (SimulatorState& state, int owner, Point2D point, function<bool(const SimulatorUnitGroup&)>* groupFilter) {
    float minDist = 10000000;
    SimulatorUnitGroup* bestGroup = nullptr;
    for (auto& g : state.groups) {
        if (g.owner == owner && (groupFilter == nullptr || (*groupFilter)(g))) {
            float dist = DistanceSquared2D(g.pos, point);
            if (dist < minDist) {
                bestGroup = &g;
                minDist = dist;
            }
        }
    }

    return bestGroup;
}

float smoothestStep(float x) {
    float x4 = (x*x)*(x*x);
    float x5 = x4*x;
    float x6 = x5*x;
    float x7 = x6*x;
    return -20*x7 + 70*x6 - 84*x5 + 35*x4;
}

float healthFraction (const SimulatorState& state, int owner) {
    float health = 0;
    float totalHealth = 0;

    for (auto& g : state.groups) {
        float h = 0;
        for (auto& u : g.units) {
            assert(u.combat.owner == g.owner);
            h += u.combat.health + u.combat.shield;
        }
        if (g.owner == owner) health += h;
        totalHealth += h;
    }

    float healthFraction = totalHealth > 0 ? health / totalHealth : 0.5f;
    healthFraction = smoothestStep(healthFraction);
    return healthFraction;
}

struct CanAttackGroup {
    vector<float> dps;

    CanAttackGroup(const vector<SimulatorUnitGroup*>& ourGroups) {
        dps = vector<float>(2);
        for (auto* group : ourGroups) {
            for (auto& unit : group->units) {
                dps[0] += calculateDPS(unit.combat.type, false);
                dps[1] += calculateDPS(unit.combat.type, true);
            }
        }
    }

    bool canAttack(const SimulatorUnitGroup& group) {
        bool hasGround = false;
        bool hasAir = false;
        for (auto& unit : group.units) {
            hasGround |= !unit.combat.is_flying;
            hasAir |= !canBeAttackedByAirWeapons(unit.combat.type);
        }

        // True if any unit can attack any other unit in the opposing team
        return (dps[0] > 0 && hasGround) || (dps[1] && hasAir);
    }
};

struct SimulatorMCTSState {
    int player = 0;
    SimulatorState state;
    int count = 0;

    SimulatorMCTSState (SimulatorState state) : state(state) {}

    pair<SimulatorMCTSState, bool> step(int action) {
        SimulatorMCTSState res = *this;
        bool validAcion = res.internalStep(action, true);
        return make_pair(move(res), validAcion);
    }

    bool internalStep(int action, bool ignoreUnintentionalNOOP=false) {
        MCTSAction mctsAction = (MCTSAction)action;
        int opponentID = (1 - player) + 1;
        int playerID = player + 1;
        switch(mctsAction) {
            case MCTSAction::ArmyMoveC1:
            case MCTSAction::ArmyMoveC2:
            case MCTSAction::ArmyMoveBase:
            case MCTSAction::NonArmyMoveBase:
            case MCTSAction::ArmyAttackClosestEnemy:
            case MCTSAction::IdleArmyAttackClosestEnemy:
            case MCTSAction::IdleNonArmyAttackClosestEnemy:
            case MCTSAction::ArmyAttackBase:
            case MCTSAction::NonArmyAttackClosestEnemy: {
                auto* groupFilter = mctsAction == MCTSAction::IdleArmyAttackClosestEnemy || mctsAction == MCTSAction::IdleNonArmyAttackClosestEnemy ? &idleGroup : nullptr;
                auto* unitFilter = mctsAction == MCTSAction::ArmyAttackBase || mctsAction == MCTSAction::ArmyAttackClosestEnemy || mctsAction == MCTSAction::IdleArmyAttackClosestEnemy || mctsAction == MCTSAction::ArmyMoveBase || mctsAction == MCTSAction::ArmyMoveC1 || mctsAction == MCTSAction::ArmyMoveC2 ? &armyUnit : &notArmyUnit;
                vector<SimulatorUnitGroup*> groups = state.select(playerID, groupFilter, unitFilter);
                if (groups.size() > 0) {
                    auto avgPos = averagePos(groups);

                    if (mctsAction == MCTSAction::ArmyMoveC1 || mctsAction == MCTSAction::ArmyMoveC2) {
                        state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, mctsAction == MCTSAction::ArmyMoveC1 ? Point2D(10, 90) : Point2D(90, 10)));
                    } else {
                        bool toOwnBase = mctsAction == MCTSAction::ArmyMoveBase || mctsAction == MCTSAction::NonArmyMoveBase;
                        CanAttackGroup canAttack(groups);
                        function<bool(const SimulatorUnitGroup&)> filter = [&](const SimulatorUnitGroup& targetGroup) { return canAttack.canAttack(targetGroup); };
                        auto* targetGroupFilter = toOwnBase || mctsAction == MCTSAction::ArmyAttackBase ? &structureGroup : &filter;
                        auto* closestEnemy = closestGroup(state, toOwnBase ? playerID : opponentID, avgPos, targetGroupFilter);

                        // TODO: Should the units get a stop order if there are no enemies?
                        if (closestEnemy != nullptr) {
                            state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, closestEnemy->pos));
                        } else {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
                break;
            }
            case MCTSAction::ArmyConsolidate:
            case MCTSAction::IdleArmyConsolidate: {
                auto* groupFilter = mctsAction == MCTSAction::IdleArmyConsolidate ? &idleGroup : nullptr;
                auto* unitFilter = &armyUnit;
                vector<SimulatorUnitGroup*> groups = state.select(playerID, groupFilter, unitFilter);
                if (groups.size() == 0) return false;
                auto avgPos = averagePos(groups);
                state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, avgPos));
                break;
            }
            case MCTSAction::None: {
                break;
            }
            case MCTSAction::ArmySuicide: {
                vector<SimulatorUnitGroup*> groups = state.select(playerID, nullptr, nullptr);
                if (groups.size() == 0) return false;
                for (auto* g : groups) {
                    for (auto& u : g->units) {
                        u.combat.health = 0;
                        u.combat.shield = 0;
                    }
                }
                state.filterDeadUnits();
                // cout << healthFraction(state, playerID) << endl;
                // assert(healthFraction(state, playerID) == 0.0f || healthFraction(state, playerID) == 0.5f);
                break;
            }
        }

        // state.simulate(state.simulator, state.time() + max(5.0f, 0.2f * state.time()));
        state.simulate(state.simulator, state.time() + 3 + 0.2f * (state.time() - state.simulator.simulationStartTime));

        player = 1 - player;
        // if (action == 0) state += 100;
        count++;
        return true;
    }

    vector<pair<int, float>> generateMoves() {
        vector<pair<int, float>> moves;
        for (int i = 0; i < 13; i++) {
            // moves.push_back({ i, min(1.0f, (i/45.0f) * 0.1f + 0.9f * 1/10.0f + 0.01f * (rand() % 10))});
            moves.push_back({ i, 0.5f });
        }
        return moves;
    }

    float rollout() const {
        // cout << state << endl;
        SimulatorMCTSState res = *this;
        res.count = 0;
        /*while(res.count < 2) {
            res.internalStep((rand() % 7));
        }*/
        for (int i = 0; i < 6; i++) {
            if (i == 3) {
                res.internalStep((rand() % 12));
            } else {
                res.state.simulate(res.state.simulator, res.state.time() + 3 + 0.2f * (res.state.time() - state.simulator.simulationStartTime));
            }
        }
        float frac = healthFraction(res.state, player + 1);
        // cout << "Simulated to " << res.state.time() << endl;
        // float frac = healthFraction(res.state, 1);
        // cout << "Eval " << player+1 << " -> " << frac << endl;
        return frac;
    }

    string to_string() const {
        stringstream ss;
        ss << "P: " << player+1 << " H: " << healthFraction(state, 1);
        return ss.str();
    }
};


void visualizeSimulator(SimulatorState& state) {
    vector<vector<float>> units;
    for (auto& group : state.groups) {
        for (auto& u : group.units) {
            units.push_back(vector<float> { group.pos.x, group.pos.y, (float)u.combat.owner, (float)u.tag, (float)u.combat.type, u.combat.health + u.combat.shield, u.combat.health_max + u.combat.shield_max });
        }
    }

    visualize_fn(units, state.time(), healthFraction(state, 1), healthFraction(state, 2));
}

void stepSimulator(SimulatorState& state, Simulator& simulator, float endTime) {
    while(state.time() < endTime) {
        state.simulate(simulator, state.time() + 1);
        visualizeSimulator(state);
    }
}

void example_simulation(Simulator& simulator, SimulatorState& state) {
    assert(state.command(1, &idleGroup, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 60))));
    assert(state.groups[0].order.type == SimulatorOrderType::Attack);
    assert(state.groups[0].order.target == Point2D(100, 60));

    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;
    visualizeSimulator(state);

    stepSimulator(state, simulator, state.time() + 2);

    assert(state.command(2, &idleGroup, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(40, 100))));

    assert(state.command(1, &idleGroup, &notArmyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(10, 60))));
    assert(state.groups[0].order.type == SimulatorOrderType::Attack);
    assert(state.groups[0].order.target == Point2D(100, 60));

    stepSimulator(state, simulator, state.time() + 6);

    assert(state.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(60, 60))));

    stepSimulator(state, simulator, state.time() + 6);

    assert(state.command(1, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));
    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;

    stepSimulator(state, simulator, state.time() + 6);

    assert(state.command(2, nullptr, &armyUnit, SimulatorOrder(SimulatorOrderType::Attack, Point2D(100, 100))));

    stepSimulator(state, simulator, state.time() + 50);
    cout << "Frac: " << healthFraction(state, 1) << " " << healthFraction(state, 2) << endl;
}

void mcts(Simulator& simulator, SimulatorState& startState) {
    srand(time(0));

    unique_ptr<State<int, SimulatorMCTSState>> state = make_unique<State<int, SimulatorMCTSState>>(SimulatorMCTSState(startState));
    for (int i = 0; i < 100000; i++) {
        mcts<int, SimulatorMCTSState>(*state);
    }

    // exit(0);
    state->print(0, 2);

    // state->getChild(6)->getChild(4)->getChild(0)->getChild(1)->getChild(4)->print(0, 2);
    // state->getChild(4)->getChild(4)->print(0, 2);

    auto* currentState = &*state;
    visualizeSimulator(currentState->internalState.state);
    int it = 0;
    while(true) {
        if (it % 2 == 0) {
            state = make_unique<State<int, SimulatorMCTSState>>(SimulatorMCTSState(currentState->internalState.state));
            simulator.simulationStartTime = state->internalState.state.time();
            for (int i = 0; i < 20000; i++) {
                mcts<int, SimulatorMCTSState>(*state);
            }
            currentState = &*state;
        }
        //auto action = ((it % 2) == 1) ? (it == 1 || it == 3 || it == 5 ? nonstd::make_optional<pair<int, State<int, SimulatorMCTSState>&>>(4, *currentState->getChild(4)) : nonstd::make_optional<pair<int, State<int, SimulatorMCTSState>&>>(0, *currentState->getChild(0))) : currentState->bestAction();
        auto action = currentState->bestAction();
        it++;
        if (action && currentState->getChild(action.value().first) != nullptr) {
            cout << "Action " << action.value().first << endl;
            currentState = &action.value().second;
            for (auto& g : currentState->internalState.state.groups) {
                cout << "Group action: " << g.order.type << " -> " << g.order.target.x << "," << g.order.target.y << " (player " << g.owner << " at " << g.pos.x << "," << g.pos.y << ")" << endl;
            }
            visualizeSimulator(currentState->internalState.state);
        } else {
            break;
        }
    }

    while(true) {
        stepSimulator(currentState->internalState.state, currentState->internalState.state.simulator, currentState->internalState.state.time() + 5);
    }
}

void test_simulator() {
    CombatPredictor combatPredictor;
    combatPredictor.init();

    Simulator simulator(&combatPredictor);
    SimulatorState state(simulator, {
        BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PROBE, 2 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 0 }, { UNIT_TYPEID::PROTOSS_STALKER, 2 }, { UNIT_TYPEID::PROTOSS_PHOENIX, 9 }, { UNIT_TYPEID::PROTOSS_IMMORTAL, 4 }}),
        BuildState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_CARRIER, 2 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 1+2 }, { UNIT_TYPEID::PROTOSS_COLOSSUS, 2 }, { UNIT_TYPEID::PROTOSS_VOIDRAY, 2 }, { UNIT_TYPEID::PROTOSS_PHOTONCANNON, 3 } }),
    },
    {
        BuildOrderState({ UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_ZEALOT, UNIT_TYPEID::PROTOSS_ZEALOT }),
        BuildOrderState({ UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_PYLON, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_ZEALOT }),
    });

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
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(100, 100), {
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_CARRIER)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(50, 10), {
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_IMMORTAL)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
        // SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_STALKER)),
    }));

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
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
        SimulatorUnit(makeUnit(2, UNIT_TYPEID::PROTOSS_PHOTONCANNON)),
    }));

    state.groups.push_back(SimulatorUnitGroup(Point2D(10, 10), {
        SimulatorUnit(makeUnit(1, UNIT_TYPEID::PROTOSS_NEXUS)),
    }));

    mcts(simulator, state);
    // example_simulation(simulator, state);
}


int main(int argc, char* argv[]) {
    initMappings();

    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        sys.path.append("bot/python")
    )");

    auto simulatorVisualizer = pybind11::module::import("simulator_visualizer");
    visualize_fn = simulatorVisualizer.attr("visualize");

    test_simulator();
}
