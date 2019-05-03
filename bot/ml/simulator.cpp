#include "simulator.h"
#include "../utilities/predicates.h"
#include "../utilities/profiler.h"
#include <array>
#include <random>
#include "simulator_context.h"

using namespace std;
using namespace sc2;

Tag simulatorUnitIndexCounter = (Tag)1 << 40;

Point2D SimulatorUnitGroup::futurePosition(float deltaTime) {
    if (order.type == SimulatorOrderType::Attack) {
        float dist = Distance2D(pos, order.target);

        if (dist > 0) {
            float minSpeed = 10000;
            for (auto& u : units) {
                minSpeed = min(minSpeed, getUnitData(u.combat.type).movement_speed);
            }

            assert(minSpeed >= 0);

            float moveDist = minSpeed * deltaTime; // TODO: What unit is speed in?
            return pos + min(1.0f, moveDist / dist) * (order.target - pos);
        }
    }
    
    return pos;
}

void removeEmptyGroups(vector<SimulatorUnitGroup>& groups) {
    auto remainingGroupIndex = groups.begin();
    for (auto& group : groups) {
        if (group.units.size() > 0) {
            if (&group != &*remainingGroupIndex) *remainingGroupIndex = move(group);
            remainingGroupIndex++;
        }
    }
    groups.erase(remainingGroupIndex, groups.end());
}

void SimulatorState::simulateGroupMovement(float endTime) {
    float deltaTime = endTime - time();
    if (deltaTime < 0) {
        cout << "Unexpected delta time " << deltaTime << " " << endTime << " " << time() << " " << &simulator << " " << shared_ptr<SimulatorContext>(simulator)->simulationStartTime << endl;
    }
    assert(deltaTime >= 0);
    for (auto& group : groups) {
        group.previousPos = group.pos;
        group.pos = group.futurePosition(deltaTime);
        if (group.pos == group.order.target) group.order.type = SimulatorOrderType::None;
        // TODO: Update unit energies
    }
}

void SimulatorState::simulateGroupCombat(float endTime) {
    // TODO: Vary depending on group sizes
    float combatThreshold = 8;
    auto simulator = shared_ptr<SimulatorContext>(this->simulator);
    auto& combatEnv = simulator->combatPredictor->getCombatEnvironment(this->states[0]->upgrades, this->states[1]->upgrades);
    vector<bool> processedGroups(groups.size());
    for (size_t gi1 = 0; gi1 < groups.size(); gi1++) {
        auto& group1 = groups[gi1];
        if (group1.owner == 1 && group1.units.size() > 0 && !processedGroups[gi1]) {
            // Note: not checking if the enemy groups are processed.
            // This allows multiple battles to happen in the same frame that involves the same enemy groups.
            // E.g. if groups A, B are close to each other and B is an enemy, but groups B and C are also close but not so close that they get included in the A-B fight.
            // Could equally well have done the check only for enemy units but not for our units.
            // Essentially we want to prevent any given pair of groups from being included in a fight more than once
            for (auto& group2 : groups) {
                if (group2.owner == 2 && group2.units.size() > 0 && DistanceSquared2D(group1.pos, group2.pos) < combatThreshold*combatThreshold) {
                    // Possible combat
                    // Collect all groups nearby
                    // Check if the groups can fight each other
                    // Simulate outcome of battle
                    auto mean = (group1.pos + group2.pos) * 0.5f;
                    vector<SimulatorUnitGroup*> nearbyGroups;
                    for (size_t gi3 = 0; gi3 < groups.size(); gi3++) {
                        auto& group3 = groups[gi3];
                        // Check if it is close enough
                        // Also make sure that owner=1 groups are never included twice in a fight
                        if (DistanceSquared2D(mean, group3.pos) < combatThreshold*combatThreshold && group3.units.size() > 0 && (group3.owner != 1 || !processedGroups[gi3])) {
                            nearbyGroups.push_back(&group3);
                            // Mark the group as processed so that we don't include it in a combat again (if it is an ally group)
                            processedGroups[gi3] = true;
                        }
                    }

                    // TODO: Handle cloak

                    array<array<float, 2>, 2> dps {{{{0, 0}}, {{0, 0}}}}; // (2, vector<float>(2));
                    array<array<bool, 2>, 2> hasAirGround {{{{false, false}}, {{false, false}}}};
                    // vector<vector<bool>> hasAirGround(2, vector<bool>(2));
                    for (auto* group : nearbyGroups) {
                        for (auto& unit : group->units) {
                            dps[group->owner - 1][0] += combatEnv.calculateDPS(1, unit.combat.type, false);
                            dps[group->owner - 1][1] += combatEnv.calculateDPS(1, unit.combat.type, true);
                            hasAirGround[group->owner - 1][0] = hasAirGround[group->owner - 1][0] | !unit.combat.is_flying;
                            hasAirGround[group->owner - 1][1] = hasAirGround[group->owner - 1][1] | canBeAttackedByAirWeapons(unit.combat.type);
                        }
                    }

                    // True if any unit can attack any other unit in the opposing team
                    bool isCombat = (dps[0][0] > 0 && hasAirGround[1][0]) || (dps[0][1] > 0 && hasAirGround[1][1]) || (dps[1][0] > 0 && hasAirGround[0][0]) || (dps[1][1] > 0 && hasAirGround[0][1]);

                    if (isCombat) {
                        // Determine which groups have moved the most recently
                        // The team that has moved the most is the attacker
                        array<float, 2> movementAmount = {{0, 0}};
                        array<float, 2> movementWeight = {{0, 0}};

                        for (auto* group : nearbyGroups) {
                            if (group->order.type != SimulatorOrderType::None) {
                                // Has target
                                float movementDist = Distance2D(group->pos, group->previousPos);
                                movementAmount[group->owner-1] += movementDist * group->units.size();
                            }
                            movementWeight[group->owner-1] += group->units.size();
                        }

                        if (movementWeight[0] > 0) movementAmount[0] /= movementWeight[0];
                        if (movementWeight[1] > 0) movementAmount[1] /= movementWeight[1];

                        int defender = movementAmount[0] < movementAmount[1] ? 1 : 2;

                        // Check the attackers movement. If it is low then both players become attackers
                        if (movementAmount[2 - defender] < 4) {
                            defender = 0;
                        }

                        float maxTime = endTime - time();
                        if (simulator->debug) {
                            cout << "Starting combat between " << nearbyGroups.size() << " groups with max time " << maxTime << endl;
                        }
                        simulator->cache.handleCombat(*this, nearbyGroups, defender, maxTime, simulator->debug);
                        #if FALSE
                        // TODO: If both armies had a fight the previous time step as well, then they should already be in position (probably)
                        // TODO: What if the combat drags on longer than to endTime? (probably common in case of harassment, it takes some time for the player to start to defend)
                        // Add a max time to predict_engage and stop. Combat will resume next simulation step.
                        // Note: have to ensure that combat is resumed properly (without the attacker having to move into range and all that)
                        CombatResult outcome = simulator->combatPredictor->predict_engage(state, false, false, nullptr, defender);

                        int index = 0;
                        for (auto* group : nearbyGroups) {
                            for (auto& u : group->units) {
                                u.combat = outcome.state.units[index];
                                index++;
                            }

                            filterDeadUnits(group);
                        }
                        #endif
                    }
                }
            }
        }
    }

    // Remove empty groups
    removeEmptyGroups(groups);
}

void SimulatorState::filterDeadUnits() {
    auto simulator = shared_ptr<SimulatorContext>(this->simulator);
    array<BuildState*, 2> newStates = {{ simulator->cache.copyState(*states[0]), simulator->cache.copyState(*states[1]) }};
    states[0] = newStates[0];
    states[1] = newStates[1];

    for (auto& group : groups) {
        // Filter out dead units from the group
        auto remainingUnitIndex = group.units.begin();
        for (auto& u : group.units) {
            if (u.combat.health > 0) {
                if (&u != &*remainingUnitIndex) *remainingUnitIndex = u;
                remainingUnitIndex++;
            } else {
                newStates[u.combat.owner - 1]->killUnits(u.combat.type, UNIT_TYPEID::INVALID, 1);
            }
        }
        group.units.erase(remainingUnitIndex, group.units.end());
    }

    newStates[0]->recalculateHash();
    newStates[1]->recalculateHash();
    removeEmptyGroups(groups);
}

void SimulatorState::filterDeadUnits(SimulatorUnitGroup* group) {
    // Filter out dead units from the group
    auto remainingUnitIndex = group->units.begin();
    for (auto& u : group->units) {
        if (u.combat.health > 0) {
            if (&u != &*remainingUnitIndex) *remainingUnitIndex = u;
            remainingUnitIndex++;
        }
    }
    group->units.erase(remainingUnitIndex, group->units.end());
}

void SimulatorState::simulateBuildOrder (float endTime) {
    assert(endTime < 100000);

    int players = states.size();
    for (int i = 0; i < players; i++) {
        int playerID = i + 1;
        function<void(const BuildEvent&)> eventCallback = [&](const BuildEvent& event) {
            switch (event.type) {
                case FinishedUnit: {
                    UNIT_TYPEID unit = abilityToUnit(event.ability);
                    // cout << "Got event " << UnitTypeToName(unit) << " " << UnitTypeToName(event.caster) << endl;
                    auto upgradedFromUnit = upgradedFrom(unit);
                    if (upgradedFromUnit != UNIT_TYPEID::INVALID) {
                        replaceUnit(playerID, upgradedFromUnit, simplifyUnitType(unit));
                    } else {
                        addUnit(playerID, simplifyUnitType(unit));
                    }
                    break;
                }
                case SpawnLarva: {
                    break;
                }
                case MuleTimeout: {
                    break;
                }
                case MakeUnitAvailable: {
                    break;
                }
                case FinishedUpgrade: {
                    break;
                }
                case WarpGateTransition: {
                    replaceUnit(playerID, UNIT_TYPEID::PROTOSS_GATEWAY, UNIT_TYPEID::PROTOSS_WARPGATE);
                    break;
                }
            }
        };

        auto simRes = shared_ptr<SimulatorContext>(simulator)->cache.simulateBuildOrder(*states[i], buildOrders[i], endTime, &eventCallback);
        states[i] = simRes.first;
        buildOrders[i].buildIndex = simRes.second.buildIndex;
        assertValidState();
        
        /*states[i].simulateBuildOrder(buildOrders[i], nullptr, false, endTime, &eventCallback);

        // If the build order is finished, simulate until the end time anyway
        if (states[i].time < endTime) states[i].simulate(endTime, &eventCallback);
        */
        assert(states[i]->time == endTime);
    }
}

/*

void BuildEvent::apply(SimulatorState& state) const {
    
}*/

bool similarOrders(const SimulatorUnitGroup& group1, const SimulatorUnitGroup& group2) {
    if (group1.order.type != group2.order.type) return false;

    float destinationMergeThreshold = 3;
    if (DistanceSquared2D(group1.order.target, group2.order.target) > destinationMergeThreshold*destinationMergeThreshold) return false;

    return true;
}

void SimulatorState::mergeGroups () {
    float mergeDistance = 6;

    for (size_t i = 0; i < groups.size(); i++) {
        auto& group1 = groups[i];
        if (group1.units.size() == 0) continue;

        bool isBuilding1 = isStationary(group1.units[0].combat.type);

        for (size_t j = i + 1; j < groups.size(); j++) {
            auto& group2 = groups[j];
            if (group2.owner != group1.owner || group2.units.size() == 0) continue;

            bool isBuilding2 = isStationary(group2.units[0].combat.type);
            if (isBuilding1 == isBuilding2 && DistanceSquared2D(group1.pos, group2.pos) < mergeDistance*mergeDistance && similarOrders(group1, group2)) {
                // Merge groups
                for (auto& u : group2.units) group1.units.push_back(u);
                // Weighted average of the group positions
                group1.pos = (group1.pos * group1.units.size() + group2.pos * group2.units.size()) / (group1.units.size() + group2.units.size());
                group2.units.clear();
            }
        }
    }

    removeEmptyGroups(groups);
}

std::pair<SimulatorUnitGroup*, SimulatorUnit*> SimulatorState::selectOne(int playerID, sc2::UNIT_TYPEID unit_type) {
    for (auto& group : groups) {
        if (group.owner == playerID) {
            for (auto& unit : group.units) {
                if (unit.combat.type == unit_type) {
                    return { &group, &unit };
                }
            }
        }
    }

    return { nullptr, nullptr };
}

vector<SimulatorUnitGroup*> SimulatorState::select(int player, std::function<bool(const SimulatorUnitGroup&)>* groupFilter, std::function<bool(const SimulatorUnit&)>* unitFilter) {
    // Note: keep track of indices instead of pointers since we are modifying the groups vector (possibly relocating it) inside the loop
    vector<int> matchingIndices;
    for (int i = (int)groups.size() - 1; i >= 0; i--) {
        auto& group = groups[i];
        if (group.owner == player && (groupFilter == nullptr || (*groupFilter)(group))) {
            if (unitFilter == nullptr) {
                matchingIndices.push_back(i);
            } else {
                SimulatorUnitGroup newGroup;
                for (int j = group.units.size() - 1; j >= 0; j--) {
                    auto& unit = group.units[j];
                    if ((*unitFilter)(unit)) {
                        newGroup.units.push_back(move(unit));
                        group.units[j] = *group.units.rbegin();
                        group.units.pop_back();
                    }
                }

                if (group.units.size() == 0) {
                    assert(newGroup.units.size() > 0);
                    // Oops, all units in the group moved apparently.
                    // Just replace the group's order
                    group.units = move(newGroup.units);
                    matchingIndices.push_back(i);
                } else if (newGroup.units.size() > 0) {
                    // Some units should get the new order
                    newGroup.pos = group.pos;
                    newGroup.owner = group.owner;
                    groups.push_back(move(newGroup));
                    matchingIndices.push_back(groups.size() - 1);
                }
            }
        }
    }
    vector<SimulatorUnitGroup*> result(matchingIndices.size());
    for (size_t i = 0; i < matchingIndices.size(); i++) {
        result[i] = &groups[matchingIndices[i]];
        assert(result[i]->owner == player);
    }

    return result;
}

void SimulatorState::command(const vector<SimulatorUnitGroup*>& selection, SimulatorOrder order, std::function<void(SimulatorUnitGroup&, SimulatorOrder)>* commandListener) {
    order.tick = tick;
    for (auto* group : selection) {
        if (commandListener != nullptr) (*commandListener)(*group, order);
        group->execute(order);
    }
}

bool SimulatorState::command(int player, std::function<bool(const SimulatorUnitGroup&)>* groupFilter, std::function<bool(const SimulatorUnit&)>* unitFilter, SimulatorOrder order) {
    auto matching = select(player, groupFilter, unitFilter);
    command(matching, order);
    return matching.size() > 0;
}

void SimulatorState::addUnit(const sc2::Unit* unit) {
    groups.push_back(SimulatorUnitGroup(unit->pos, { SimulatorUnit(CombatUnit(*unit)) }));
    auto& group = *groups.rbegin();
    group.units[0].tag = unit->tag;

    // TODO: Add order also when just attacking an enemy
    if (unit->orders.size() > 0 && Point2D(0, 0) != unit->orders[0].target_pos) {
        group.order = SimulatorOrder(SimulatorOrderType::Attack, unit->orders[0].target_pos);
        group.order.tick = -1000;
    }
}

void SimulatorState::addUnit(CombatUnit unit, sc2::Point2D pos) {
    groups.push_back(SimulatorUnitGroup(pos, { SimulatorUnit(unit) }));
}

void SimulatorState::addUnit(int owner, sc2::UNIT_TYPEID unit_type) {
    // Add around some buildings
    SimulatorUnitGroup* randomGroup = nullptr;
    int weight = 0;

    // Base the seed on some function of the state pointers
    // This ensures that the simulation is deterministic (with caching)
    // even when it includes randomness
    
    uint64_t seed = (uint64_t)states[0]->immutableHash() + (uint64_t)states[1]->immutableHash();
    default_random_engine rnd(seed);

    // Reservoir sampling
    for (auto& g : groups) {
        if (g.owner == owner && isStructure(g.units[0].combat.type)) {
            // Yes! A structure group
            weight++;
            uniform_int_distribution<int> dist(0, weight - 1);
            if (dist(rnd) == 0) {
                randomGroup = &g;
            }
        }
    }

    uniform_real_distribution<double> offsetDist(-0.5f, 0.5f);
    float dx = offsetDist(rnd);
    float dy = offsetDist(rnd);
    auto pos = (randomGroup != nullptr ? randomGroup->pos : shared_ptr<SimulatorContext>(simulator)->defaultPositions[owner]) + Point2D(dx * 5, dy * 5);
    addUnit(makeUnit(owner, unit_type), pos);
}

void SimulatorState::replaceUnit(int owner, sc2::UNIT_TYPEID unit_type, sc2::UNIT_TYPEID replacement) {
    for (auto& g : groups) {
        if (g.owner == owner) {
            for (auto& u : g.units) {
                if (u.combat.type == unit_type) {
                    // TODO: Preserve health fraction
                    u.combat = makeUnit(owner, replacement);
                    return;
                }
            }
        }
    }

    cerr << "Could not replace unit " << UnitTypeToName(unit_type) << " with " << UnitTypeToName(replacement) << " for player " << owner << endl;
    assert(false);
}

Stopwatch w1(false);
Stopwatch w2(false);
Stopwatch w3(false);
Stopwatch w4(false);
float t1, t2, t3, t4;

int c = 0;

void SimulatorState::simulate (float endTime) {
    if (simulator.expired()) {
        cerr << "Simulator no longer exists" << endl;
        assert(false);
    }
    if (endTime < time()) throw std::out_of_range("endTime");

    tick++;

    // Move unit groups
    // Determine if any combat should happen
    // Simulate build state
    // Figure out which new units were created during this time
    // Merge groups
    // Determine if any combat should happen

    w1.start();
    float midTime = (time() + endTime) * 0.5f;

    bool checkState = (c % 10000) == 0;

    w3.start();
    simulateGroupMovement(midTime);
    if (checkState) assertValidState();
    w3.stop();
    t3 += w3.millis();

    w4.start();
    simulateGroupCombat(midTime);
    if (checkState) assertValidState();
    w4.stop();
    t4 += w4.millis();
    
    w2.start();
    simulateBuildOrder(midTime);
    w2.stop();
    t2 += w2.millis();
    if (checkState) assertValidState();

    w3.start();
    simulateGroupMovement(endTime);
    if (checkState) assertValidState();
    w3.stop();
    t3 += w3.millis();

    w4.start();
    simulateGroupCombat(endTime);
    if (checkState) assertValidState();
    w4.stop();
    t4 += w4.millis();

    w2.start();
    simulateBuildOrder(endTime);
    w2.stop();
    t2 += w2.millis();
    if (checkState) assertValidState();

    // New units?

    mergeGroups();
    if (checkState) assertValidState();
    w1.stop();
    t1 += w1.millis();

    // Simulate group combat again
    assert(time() == endTime);
    c++;
    if ((c % 10000) == 0) {
        // cout << "Times " << t1 << " " << t2 << " " << t3 << " " << t4 << endl;
    }
    
}

void SimulatorState::assertValidState () {
    // return;
    for (int k = 0; k < 2; k++) {
        const BuildState& buildState = *states[k];
        map<UNIT_TYPEID, int> unitCounts;
        map<UNIT_TYPEID, int> unitCounts2;
        for (auto& g : groups) {
            if (g.owner == k + 1) {
                for (auto& u : g.units) {
                    unitCounts[u.combat.type] += 1;
                    unitCounts2[u.combat.type] += 0;
                }
            }
        }
        // Just make sure they are listed
        for (auto& u : buildState.units) {
            unitCounts[u.type] += 0;
            unitCounts2[u.type] += u.units;
        }

        for (auto p : unitCounts) {
            if (unitCounts2[p.first] != p.second) {
                cerr << "Mismatch in unit counts " << UnitTypeToName(p.first) << " " << unitCounts2[p.first] << " " << p.second << endl;
                cerr << "For player " << k << endl;
                cerr << time() << endl;
                for (auto p : unitCounts2) {
                    cout << "Has " << UnitTypeToName(p.first) << " " << p.second << endl;
                }
                for (auto p : unitCounts) {
                    cout << "Expected " << UnitTypeToName(p.first) << " " << p.second << endl;
                }
                assert(false);
            }
        }
    }
}

SimulatorState::SimulatorState (std::shared_ptr<SimulatorContext> simulator, const std::vector<BuildState>& states, const std::vector<BuildOrderState>& buildOrders) : simulator(simulator), buildOrders(buildOrders) {
    assert(states.size() == 2);
    assert(buildOrders.size() == 2);
    this->states = {{ simulator->cache.copyState(states[0]), simulator->cache.copyState(states[1]) }};
}
