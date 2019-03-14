#include "simulator.h"
#include "../utilities/predicates.h"
#include <array>

using namespace std;
using namespace sc2;

int simulatorUnitIndexCounter = 0;

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

void SimulatorState::simulateGroupMovement(Simulator& simulator, float endTime) {
    float deltaTime = endTime - states[0].time;
    assert(deltaTime >= 0);
    for (auto& group : groups) {
        group.previousPos = group.pos;
        group.pos = group.futurePosition(deltaTime);
        // TODO: Update unit energies
    }
}

void SimulatorState::simulateGroupCombat(Simulator& simulator, float endTime) {
    // TODO: Vary depending on group sizes
    float combatThreshold = 8;

    for (auto& group1 : groups) {
        if (group1.owner == 1 && group1.units.size() > 0) {
            for (auto& group2 : groups) {
                if (group2.owner == 2 && group2.units.size() > 0 && DistanceSquared2D(group1.pos, group2.pos) < combatThreshold*combatThreshold) {
                    // Possible combat
                    // Collect all groups nearby
                    // Check if the groups can fight each other
                    // Simulate outcome of battle
                    auto mean = (group1.pos + group2.pos) * 0.5f;
                    vector<SimulatorUnitGroup*> nearbyGroups;
                    for (auto& group3 : groups) {
                        if (DistanceSquared2D(mean, group3.pos) < combatThreshold*combatThreshold && group3.units.size() > 0) {
                            nearbyGroups.push_back(&group3);
                        }
                    }

                    // TODO: Handle cloak

                    vector<vector<float>> dps (2, vector<float>(2));
                    vector<vector<bool>> hasAirGround(2, vector<bool>(2));
                    for (auto* group : nearbyGroups) {
                        for (auto& unit : group->units) {
                            dps[group->owner - 1][0] += calculateDPS(unit.combat.type, false);
                            dps[group->owner - 1][1] += calculateDPS(unit.combat.type, true);
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

                        CombatState state;
                        for (auto* group : nearbyGroups) {
                            if (group->order.type != SimulatorOrderType::None) {
                                // Has target
                                float movementDist = Distance2D(group->pos, group->previousPos);
                                movementAmount[group->owner-1] += movementDist * group->units.size();
                            }
                            movementWeight[group->owner-1] += group->units.size();
                            for (auto& u : group->units) state.units.push_back(u.combat);
                        }

                        if (movementWeight[0] > 0) movementAmount[0] /= movementWeight[0];
                        if (movementWeight[1] > 0) movementAmount[1] /= movementWeight[1];

                        int defender = movementAmount[0] < movementAmount[1] ? 1 : 2;

                        // TODO: Swap owners, it is assumed that owner == 1 is the defender
                        // TODO: What if the combat drags on longer than to endTime? (probably common in case of harassment, it takes some time for the player to start to defend)
                        // Add a max time to predict_engage and stop. Combat will resume next simulation step.
                        // Note: have to ensure that combat is resumed properly (without the attacker having to move into range and all that)
                        CombatResult outcome = simulator.combatPredictor->predict_engage(state, false, false, nullptr, defender);

                        int index = 0;
                        for (auto* group : nearbyGroups) {
                            for (auto& u : group->units) {
                                u.combat = outcome.state.units[index];
                                index++;
                            }

                            filterDeadUnits(group);
                        }
                    }
                }
            }
        }
    }

    // Remove empty groups
    removeEmptyGroups(groups);
}

void SimulatorState::filterDeadUnits() {
    for (auto& group : groups) {
        // Filter out dead units from the group
        auto remainingUnitIndex = group.units.begin();
        for (auto& u : group.units) {
            if (u.combat.health > 0) {
                if (&u != &*remainingUnitIndex) *remainingUnitIndex = u;
                remainingUnitIndex++;
            } else {
                // TODO: Handle addons?
                states[u.combat.owner - 1].killUnits(u.combat.type, UNIT_TYPEID::INVALID, 1);
            }
        }
        group.units.erase(remainingUnitIndex, group.units.end());
    }
    removeEmptyGroups(groups);
}

void SimulatorState::filterDeadUnits(SimulatorUnitGroup* group) {
    // Filter out dead units from the group
    auto remainingUnitIndex = group->units.begin();
    for (auto& u : group->units) {
        if (u.combat.health > 0) {
            if (&u != &*remainingUnitIndex) *remainingUnitIndex = u;
            remainingUnitIndex++;
        } else {
            // TODO: Handle addons?
            states[u.combat.owner - 1].killUnits(u.combat.type, UNIT_TYPEID::INVALID, 1);
        }
    }
    group->units.erase(remainingUnitIndex, group->units.end());
}

void SimulatorState::simulateBuildOrder (Simulator& simulator, float endTime) {
    int players = states.size();
    for (int i = 0; i < players; i++) {
        int playerID = i + 1;
        function<void(const BuildEvent&)> eventCallback = [&](const BuildEvent& event) {
            switch (event.type) {
                case FinishedUnit: {
                    UNIT_TYPEID unit = abilityToUnit(event.ability);

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
            }
        };
        states[i].simulateBuildOrder(buildOrders[i], nullptr, false, endTime, &eventCallback);

        // If the build order is finished, simulate until the end time anyway
        if (states[i].time < endTime) states[i].simulate(endTime, &eventCallback);
        assert(states[i].time == endTime);
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

void SimulatorState::mergeGroups (Simulator& simulator) {
    float mergeDistance = 6;

    for (int i = 0; i < groups.size(); i++) {
        auto& group1 = groups[i];
        if (group1.units.size() == 0) continue;

        bool isBuilding1 = isStationary(group1.units[0].combat.type);

        for (int j = i + 1; j < groups.size(); j++) {
            auto& group2 = groups[j];
            if (group2.units.size() == 0) continue;

            if (group2.owner == group1.owner) {
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
    }

    removeEmptyGroups(groups);
}

vector<SimulatorUnitGroup*> SimulatorState::select(int player, std::function<bool(const SimulatorUnitGroup&)>* groupFilter, std::function<bool(const SimulatorUnit&)>* unitFilter) {
    // Note: keep track of indices instead of pointers since we are modifying the groups vector (possibly relocating it) inside the loop
    vector<int> matchingIndices;
    for (int i = groups.size() - 1; i >= 0; i--) {
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
    for (int i = 0; i < matchingIndices.size(); i++) {
        result[i] = &groups[matchingIndices[i]];
        assert(result[i]->owner == player);
    }

    return result;
}

void SimulatorState::command(const vector<SimulatorUnitGroup*>& selection, SimulatorOrder order) {
    for (auto* group : selection) {
        group->execute(order);
    }
}

bool SimulatorState::command(int player, std::function<bool(const SimulatorUnitGroup&)>* groupFilter, std::function<bool(const SimulatorUnit&)>* unitFilter, SimulatorOrder order) {
    auto matching = select(player, groupFilter, unitFilter);
    command(matching, order);
    return matching.size() > 0;
}

void SimulatorState::addUnit(int owner, sc2::UNIT_TYPEID unit_type) {
    // Add around some buildings
    SimulatorUnitGroup* randomGroup = nullptr;
    int weight = 0;

    // Reservoir sampling
    for (auto& g : groups) {
        if (g.owner == owner && isStructure(g.units[0].combat.type)) {
            // Yes! A structure group
            weight++;
            if ((rand() % weight) == 0) {
                randomGroup = &g;
            }
        }
    }

    if (randomGroup == nullptr) {
        // ???
        // cerr << "Could not build unit " << UnitTypeToName(unit_type) << " because there are no buildings to built it around" << endl;
    } else {
        float dx = (rand() % 10000)/10000.0f - 0.5f;
        float dy = (rand() % 10000)/10000.0f - 0.5f;
        auto pos = randomGroup->pos + Point2D(dx * 5, dy * 5);
        groups.push_back(SimulatorUnitGroup(pos, { SimulatorUnit(makeUnit(owner, unit_type)) }));
    }
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
}

void SimulatorState::simulate (Simulator& simulator, float endTime) {
    if (endTime < time()) throw std::out_of_range("endTime");

    // Move unit groups
    // Determine if any combat should happen
    // Simulate build state
    // Figure out which new units were created during this time
    // Merge groups
    // Determine if any combat should happen

    float midTime = (time() + endTime) * 0.5f;
    simulateGroupMovement(simulator, midTime);
    simulateGroupCombat(simulator, midTime);
    simulateBuildOrder(simulator, midTime);

    simulateGroupMovement(simulator, endTime);
    simulateGroupCombat(simulator, endTime);
    simulateBuildOrder(simulator, endTime);

    // New units?

    mergeGroups(simulator);

    // Simulate group combat again
}