#include "../ml/simulator.h"
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/utilities/mappings.h>
#include <functional>
#include <random>
#include <sstream>
#include <iostream>
#include "../mcts/mcts.h"
#include "../ml/mcts_sc2.h"
#include "simulator_context.h"

using namespace std;
using namespace sc2;

function<bool(const SimulatorUnitGroup&)> idleGroup = [](const SimulatorUnitGroup& group) { return group.order.type == SimulatorOrderType::None; };
function<bool(const SimulatorUnitGroup&)> structureGroup = [](const SimulatorUnitGroup& group) { return isStructure(group.units[0].combat.type); };
function<bool(const SimulatorUnit&)> armyUnit = [](const SimulatorUnit& unit) { return isArmy(unit.combat.type); };
function<bool(const SimulatorUnit&)> notArmyUnit = [](const SimulatorUnit& unit) { return !isArmy(unit.combat.type); };
function<bool(const SimulatorUnit&)> structureUnit = [](const SimulatorUnit& unit) { return isStructure(unit.combat.type); };

Point2D averagePos(vector<SimulatorUnitGroup*> groups) {
    Point2D p;
    float weight = 0;
    for (auto* g : groups) {
        float w = sqrt(g->units.size());
        p += g->pos * w;
        weight += w;
    }
    return p / (weight + 0.00001f);
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
            // h += u.combat.health + u.combat.shield;
            auto& data = getUnitData(u.combat.type);
            h += (data.mineral_cost + data.vespene_cost) * (0.5f + 0.5f * (u.combat.health + u.combat.shield) / (maxHealth(u.combat.type) + maxShield(u.combat.type)));
            if (!isfinite(h)) {
                cerr << "Weird health for unit " << getUnitData(u.combat.type).name << " " << h << " " << u.combat.health << " " << u.combat.shield << " " << maxHealth(u.combat.type) << " " << maxShield(u.combat.type) << endl;
            }
        }

        if (g.owner == owner && structureGroup(g)) {
            h *= 3;
        }

        if (g.owner == owner) health += h;
        totalHealth += h;
    }

    float healthFraction = totalHealth > 0.0001f ? health / totalHealth : 0.5f;
    healthFraction = smoothestStep(healthFraction);

    float mult = 100 / (100 + state.time());
    healthFraction = (healthFraction - 0.5f) * mult + 0.5f;
    assert(isfinite(healthFraction));
    return healthFraction;
}

struct CanAttackGroup {
    vector<float> dps;

    CanAttackGroup(const CombatPredictor& combatPredictor, const vector<SimulatorUnitGroup*>& ourGroups) {
        dps = vector<float>(2);
        for (auto* group : ourGroups) {
            for (auto& unit : group->units) {
                dps[0] += combatPredictor.defaultCombatEnvironment.calculateDPS(1, unit.combat.type, false);
                dps[1] += combatPredictor.defaultCombatEnvironment.calculateDPS(1, unit.combat.type, true);
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

bool isValidDestination (vector<SimulatorUnitGroup*> groups, Point2D dest, int currentTick) {
    int yes = 0;
    int totalVotes = 0;
    for (auto* g : groups) {
        int numUnits = g->units.size();
        totalVotes += numUnits;

        if (g->order.tick < currentTick - 2) {
            yes += numUnits;
            continue;
        }

        if (Point2D(0,0) == g->order.target) {
            yes += numUnits;
            continue;
        }

        float r1 = DistanceSquared2D(g->order.target, g->pos);

        // Veeery close to the destination, treat the order as finished
        if (r1 <= 3*3) {
            yes += numUnits;
            continue;
        }

        // Ok if the destination is sort of in the same direction as the current destination
        if (DistanceSquared2D(g->order.target, dest) <= r1) {
            yes += numUnits;
            continue;
        }

        // Also ok if the destination is very close to the group (note: 0.25 = 0.5^2)
        if (DistanceSquared2D(g->pos, dest) <= r1 * 0.25f) {
            yes += numUnits;
            continue;
        }
    }

    return yes*2 >= totalVotes;
}

bool SimulatorMCTSState::executeAction(MCTSAction action, std::function<void(SimulatorUnitGroup&, SimulatorOrder)>* commandListener) {
    bool sameActionAsLastTime = (lastActions[player] == (MCTSAction)action);
    lastActions[player] = (MCTSAction)action;
    int opponentID = (1 - player) + 1;
    int playerID = player + 1;
    switch(action) {
        case MCTSAction::ArmyMoveC1:
        case MCTSAction::ArmyMoveC2:
        case MCTSAction::ArmyMoveC3: {
            auto* unitFilter = &armyUnit;
            vector<SimulatorUnitGroup*> groups = state.select(playerID, nullptr, unitFilter);
            if (groups.size() > 0) {
                auto avgPos = averagePos(groups);
                auto simulator = shared_ptr<SimulatorContext>(state.simulator);

                int destinationIndex = (int)action - (int)MCTSAction::ArmyMoveC1;
                Point2D destination = simulator->extraDestinations[player][destinationIndex];

                // Point2D destination = action == MCTSAction::ArmyMoveC1 ? Point2D(168/2, 168/2) : Point2D(168/2, 160);
                if (!sameActionAsLastTime && !isValidDestination(groups, destination, state.tick)) return false;
                state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, destination), commandListener);
            }
            break;
        }
        case MCTSAction::ArmyMoveBase:
        case MCTSAction::NonArmyMoveBase:
        case MCTSAction::ArmyAttackClosestEnemy:
        case MCTSAction::IdleArmyAttackClosestEnemy:
        case MCTSAction::IdleNonArmyAttackClosestEnemy:
        case MCTSAction::ArmyAttackBase:
        case MCTSAction::NonArmyAttackClosestEnemy: {
            auto* groupFilter = action == MCTSAction::IdleArmyAttackClosestEnemy || action == MCTSAction::IdleNonArmyAttackClosestEnemy ? &idleGroup : nullptr;
            auto* unitFilter = action == MCTSAction::ArmyAttackBase || action == MCTSAction::ArmyAttackClosestEnemy || action == MCTSAction::IdleArmyAttackClosestEnemy || action == MCTSAction::ArmyMoveBase || action == MCTSAction::ArmyMoveC1 || action == MCTSAction::ArmyMoveC2 || action == MCTSAction::ArmyMoveC3 ? &armyUnit : &notArmyUnit;
            vector<SimulatorUnitGroup*> groups = state.select(playerID, groupFilter, unitFilter);
            if (groups.size() > 0) {
                auto avgPos = averagePos(groups);
                auto simulator = shared_ptr<SimulatorContext>(state.simulator);

                bool toOwnBase = action == MCTSAction::ArmyMoveBase || action == MCTSAction::NonArmyMoveBase;
                CanAttackGroup canAttack(*simulator->combatPredictor, groups);
                function<bool(const SimulatorUnitGroup&)> filter = [&](const SimulatorUnitGroup& targetGroup) { return canAttack.canAttack(targetGroup); };
                auto* targetGroupFilter = toOwnBase || action == MCTSAction::ArmyAttackBase ? &structureGroup : &filter;
                auto* closestEnemy = closestGroup(state, toOwnBase ? playerID : opponentID, avgPos, targetGroupFilter);

                // TODO: Should the units get a stop order if there are no enemies?
                if (closestEnemy != nullptr) {
                    if (!sameActionAsLastTime && !isValidDestination(groups, closestEnemy->pos, state.tick)) return false;

                    state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, closestEnemy->pos), commandListener);
                } else {
                    return false;
                }
            } else {
                return false;
            }
            break;
        }
        case MCTSAction::Reinforce: {
            float bestScore = 0;
            SimulatorUnitGroup* bestGroup = nullptr;
            for (auto& group : state.groups) {
                if (group.owner == playerID && !structureGroup(group)) {
                    int numArmyUnits = 0;
                    for (auto& u : group.units) numArmyUnits += isArmy(u.combat.type);
                    float score = numArmyUnits;
                    if (score > bestScore) {
                        bestScore = score;
                        bestGroup = &group;
                    }
                }
            }
            if (bestGroup == nullptr) return false;
            auto avgPos = bestGroup->pos;

            vector<SimulatorUnitGroup*> groups = state.select(playerID, nullptr, &armyUnit);
            // Note: probably invalidated by the select call
            bestGroup = nullptr;

            if (!sameActionAsLastTime && !isValidDestination(groups, avgPos, state.tick)) return false;

            state.command(groups, SimulatorOrder(SimulatorOrderType::Attack, avgPos), commandListener);
            break;
        }
        case MCTSAction::ArmyConsolidate:
        case MCTSAction::IdleArmyConsolidate: {
            auto* groupFilter = action == MCTSAction::IdleArmyConsolidate ? &idleGroup : nullptr;
            auto* unitFilter = &armyUnit;
            vector<SimulatorUnitGroup*> groups = state.select(playerID, groupFilter, unitFilter);
            if (groups.size() == 0) return false;
            auto avgPos = averagePos(groups);
            if (!sameActionAsLastTime && !isValidDestination(groups, avgPos, state.tick)) return false;
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
        default: {
            assert(false);
            break;
        }
    }

    return true;
}

static float t1, t2, t3, t4, t5;
static int counter = 0;

bool SimulatorMCTSState::internalStep(int action, bool ignoreUnintentionalNOOP) {
    auto simulator = shared_ptr<SimulatorContext>(state.simulator);
    Stopwatch w1;
    if (!executeAction((MCTSAction)action)) {
        if (ignoreUnintentionalNOOP) {
            w1.stop();
            t5 += w1.millis();
            return false;
        }
    }

    {
        // Repeat the last opponent action
        player = 1 - player;
        executeAction(lastActions[player]);
        player = 1 - player;
    }

    w1.stop();
    t5 += w1.millis();

    // state.simulate(state.simulator, state.time() + max(5.0f, 0.2f * state.time()));
    assert(state.time() >= simulator->simulationStartTime);
    if ((state.time() + 3 + 0.2f * (state.time() - simulator->simulationStartTime)) > 10000) {
        // cerr << "Too large time " << state.time() << " " << simulator.simulationStartTime << endl;
        // assert(false);
    }
    state.simulate(state.time() + 3 + 0.2f * (state.time() - simulator->simulationStartTime));

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

    for (int i = 0; i < 9; i++) {
        // moves.push_back({ i, min(1.0f, (i/45.0f) * 0.1f + 0.9f * 1/10.0f + 0.01f * (rand() % 10))});
        moves.push_back({ i, 0.5f });
    }
    return moves;
}

array<float, 2> SimulatorMCTSState::rollout() const {
    auto simulator = shared_ptr<SimulatorContext>(state.simulator);
    Stopwatch w1;
    int w = isWin();
    w1.stop();
    t1 += w1.millis();
    // if (w == -1) return {{ 0.5f, 0.5f }};
    // if (w != 0) return {{ w == 1 ? 1.0f : 0.0f, w == 2 ? 1.0f : 0.0f }};
    if (w != 0) return state.rewards;

    Stopwatch w2;
    // cout << state << endl;
    SimulatorMCTSState res = *this;
    res.count = 0;
    w2.stop();
    t2 += w2.millis();

    Stopwatch w3;
    for (int i = 0; i < 6; i++) {
        if (i == 3) {
            res.internalStep((rand() % 9), false);
        } else {
            res.internalStep((int)MCTSAction::None, false);
        }
    }
    w3.stop();
    t3 += w3.millis();

    Stopwatch w4;
    w4.stop();
    t4 += w4.millis();

    counter++;
    if ((counter % 10000) == 0) {
        // cout << "Rollout stats " << t1 << " " << t2 << " " << t3 << " " << t4 << " " << t5 << endl;
    }
    // cout << "Simulated to " << res.state.time() << endl;
    // float frac = healthFraction(res.state, 1);
    // cout << "Eval " << player+1 << " -> " << frac << endl;
    // return {{ healthFraction(res.state, 1), healthFraction(res.state, 2) }};
    return res.state.rewards;
}

string SimulatorMCTSState::to_string() const {
    stringstream ss;
    ss << "P: " << player+1 << " H: " << healthFraction(state, 1);
    return ss.str();
}

MCTSSearchSC2 findBestActions(SimulatorState& startState, int startingPlayerIndex) {
    assert(startingPlayerIndex == 0 || startingPlayerIndex == 1);
    auto simulator = shared_ptr<SimulatorContext>(startState.simulator);
    unique_ptr<MCTSState<int, SimulatorMCTSState>> state = make_unique<MCTSState<int, SimulatorMCTSState>>(SimulatorMCTSState(startState, startingPlayerIndex));
    simulator->simulationStartTime = startState.time();

    auto search = make_unique<MCTSSearch<int, SimulatorMCTSState>>(startState);
    search->search(30000);

    return MCTSSearchSC2(search, simulator);

    /*auto* s = &*state;
    int depth = 0;
    while(true) {
        if (s->bestAction()) {
            s = &s->bestAction().value().second;
        } else {
            break;
        }
        depth++;
    }*/
    // cout << "SEARCH DEPTH " << depth << endl;
    // return state;
}
