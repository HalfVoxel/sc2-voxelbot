#include "../ml/simulator.h"
#include "../utilities/predicates.h"
#include "../utilities/mappings.h"
#include <functional>
#include <random>
#include <sstream>
#include <iostream>
#include "../mcts/mcts.h"
#include "../ml/mcts_sc2.h"

using namespace std;
using namespace sc2;

function<bool(const SimulatorUnitGroup&)> idleGroup = [](const SimulatorUnitGroup& group) { return group.order.type == SimulatorOrderType::None; };
function<bool(const SimulatorUnitGroup&)> structureGroup = [](const SimulatorUnitGroup& group) { return isStructure(group.units[0].combat.type); };
function<bool(const SimulatorUnit&)> armyUnit = [](const SimulatorUnit& unit) { return isArmy(unit.combat.type); };
function<bool(const SimulatorUnit&)> notArmyUnit = [](const SimulatorUnit& unit) { return !isArmy(unit.combat.type); };
function<bool(const SimulatorUnit&)> structureUnit = [](const SimulatorUnit& unit) { return isStructure(unit.combat.type); };

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

pair<SimulatorMCTSState, bool> SimulatorMCTSState::step(int action) {
    SimulatorMCTSState res = *this;
    bool validAcion = res.internalStep(action, true);
    return make_pair(move(res), validAcion);
}

bool SimulatorMCTSState::executeAction(MCTSAction action, std::function<void(SimulatorUnitGroup&, SimulatorOrder)>* commandListener) {
    int opponentID = (1 - player) + 1;
    int playerID = player + 1;
    switch(action) {
        case MCTSAction::ArmyMoveC1:
        case MCTSAction::ArmyMoveC2:
        case MCTSAction::ArmyMoveBase:
        case MCTSAction::NonArmyMoveBase:
        case MCTSAction::ArmyAttackClosestEnemy:
        case MCTSAction::IdleArmyAttackClosestEnemy:
        case MCTSAction::IdleNonArmyAttackClosestEnemy:
        case MCTSAction::ArmyAttackBase:
        case MCTSAction::NonArmyAttackClosestEnemy: {
            auto* groupFilter = action == MCTSAction::IdleArmyAttackClosestEnemy || action == MCTSAction::IdleNonArmyAttackClosestEnemy ? &idleGroup : nullptr;
            auto* unitFilter = action == MCTSAction::ArmyAttackBase || action == MCTSAction::ArmyAttackClosestEnemy || action == MCTSAction::IdleArmyAttackClosestEnemy || action == MCTSAction::ArmyMoveBase || action == MCTSAction::ArmyMoveC1 || action == MCTSAction::ArmyMoveC2 ? &armyUnit : &notArmyUnit;
            vector<SimulatorUnitGroup*> groups = state.select(playerID, groupFilter, unitFilter);
            if (groups.size() > 0) {
                auto avgPos = averagePos(groups);

                if (action == MCTSAction::ArmyMoveC1 || action == MCTSAction::ArmyMoveC2) {
                    state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, action == MCTSAction::ArmyMoveC1 ? Point2D(10, 90) : Point2D(90, 10)), commandListener);
                } else {
                    bool toOwnBase = action == MCTSAction::ArmyMoveBase || action == MCTSAction::NonArmyMoveBase;
                    CanAttackGroup canAttack(groups);
                    function<bool(const SimulatorUnitGroup&)> filter = [&](const SimulatorUnitGroup& targetGroup) { return canAttack.canAttack(targetGroup); };
                    auto* targetGroupFilter = toOwnBase || action == MCTSAction::ArmyAttackBase ? &structureGroup : &filter;
                    auto* closestEnemy = closestGroup(state, toOwnBase ? playerID : opponentID, avgPos, targetGroupFilter);

                    // TODO: Should the units get a stop order if there are no enemies?
                    if (closestEnemy != nullptr) {
                        state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, closestEnemy->pos), commandListener);
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
            auto* groupFilter = action == MCTSAction::IdleArmyConsolidate ? &idleGroup : nullptr;
            auto* unitFilter = &armyUnit;
            vector<SimulatorUnitGroup*> groups = state.select(playerID, groupFilter, unitFilter);
            if (groups.size() == 0) return false;
            auto avgPos = averagePos(groups);
            state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, avgPos), commandListener);
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

    return true;
}

bool SimulatorMCTSState::internalStep(int action, bool ignoreUnintentionalNOOP) {
    if (!executeAction((MCTSAction)action)) return false;

    // state.simulate(state.simulator, state.time() + max(5.0f, 0.2f * state.time()));
    assert(state.time() >= state.simulator.simulationStartTime);
    if ((state.time() + 3 + 0.2f * (state.time() - state.simulator.simulationStartTime)) > 10000) {
        cerr << "Too large time " << state.time() << " " << state.simulator.simulationStartTime << endl;
        assert(false);
    }
    state.simulate(state.simulator, state.time() + 3 + 0.2f * (state.time() - state.simulator.simulationStartTime));

    player = 1 - player;
    // if (action == 0) state += 100;
    count++;
    return true;
}

int SimulatorMCTSState::isWin() const {
    array<bool, 2> hasStructure = {{ false, false }};
    for (auto& g : state.groups) {
        for (auto& u : g.units) {
            if (isStructure(u.combat.type)) {
                hasStructure[u.combat.owner - 1] = true;
            }
        }
    }

    if (!hasStructure[0] && !hasStructure[1]) {
        // Tie
        return -1;
    }

    // If player 1 has lost, then player 2 wins
    if (!hasStructure[0]) return 2;
    // If player 2 has lost, then player 1 wins
    if (!hasStructure[1]) return 1;

    // No win yet
    return 0;
}

vector<pair<int, float>> SimulatorMCTSState::generateMoves() {
    vector<pair<int, float>> moves;
    if (isWin() != 0) return moves;

    for (int i = 0; i < 10; i++) {
        // moves.push_back({ i, min(1.0f, (i/45.0f) * 0.1f + 0.9f * 1/10.0f + 0.01f * (rand() % 10))});
        moves.push_back({ i, 0.5f });
    }
    return moves;
}

float SimulatorMCTSState::rollout() const {
    int w = isWin();
    if (w == -1) return 0.5f;
    if (w != 0) return player + 1 == w ? 1 : 0;

    // cout << state << endl;
    SimulatorMCTSState res = *this;
    res.count = 0;
    /*while(res.count < 2) {
        res.internalStep((rand() % 7));
    }*/
    for (int i = 0; i < 6; i++) {
        if (i == 3) {
            res.internalStep((rand() % 9));
        } else {
            assert(res.state.time() >= state.simulator.simulationStartTime);
            float newEndTime = res.state.time() + 3 + 0.2f * (res.state.time() - state.simulator.simulationStartTime);
            if (newEndTime > 10000) {
                cerr << "Too large time2 " << res.state.time() << " " << res.state.simulator.simulationStartTime << endl;
                assert(false);
            }
            res.state.simulate(res.state.simulator, newEndTime);
        }
    }
    float frac = healthFraction(res.state, player + 1);
    // cout << "Simulated to " << res.state.time() << endl;
    // float frac = healthFraction(res.state, 1);
    // cout << "Eval " << player+1 << " -> " << frac << endl;
    return frac;
}

string SimulatorMCTSState::to_string() const {
    stringstream ss;
    ss << "P: " << player+1 << " H: " << healthFraction(state, 1);
    return ss.str();
}

unique_ptr<State<int, SimulatorMCTSState>> findBestActions(SimulatorState& startState) {
    unique_ptr<State<int, SimulatorMCTSState>> state = make_unique<State<int, SimulatorMCTSState>>(SimulatorMCTSState(startState));
    startState.simulator.simulationStartTime = startState.time();

    for (int i = 0; i < 30000; i++) {
        mcts<int, SimulatorMCTSState>(*state);
    }

    auto* s = &*state;
    int depth = 0;
    while(true) {
        if (s->bestAction()) {
            s = &s->bestAction().value().second;
        } else {
            break;
        }
        depth++;
    }
    cout << "SEARCH DEPTH " << depth << endl;
    return state;
}
