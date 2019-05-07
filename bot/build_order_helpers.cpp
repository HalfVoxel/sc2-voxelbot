#include "build_order_helpers.h"
#include "utilities/mappings.h"
#include "behaviortree/TacticalNodes.h"
#include "behaviortree/BehaviorTree.h"
#include "utilities/predicates.h"
#include "utilities/stdutils.h"
#include "unit_lists.h"
#include <sstream>

using namespace std;
using namespace sc2;
using namespace BOT;

vector<const Unit*> recentlySeedGasHarvesters;

/** Returns how many harvesters are currently hidden because they are inside Assimilators/Refineries/Extractors */
int countMissingVespeneWorkers (const vector<const Unit*>& ourUnits) {
    int currentLoop = agent->Observation()->GetGameLoop();
    int numHiddenHarvesters = 0;
    for (int i = recentlySeedGasHarvesters.size() - 1; i >= 0; i--) {
        const auto* u = recentlySeedGasHarvesters[i];
        // Theoretically the workers are inside the assimilator for 32 ticks
        // See https://liquipedia.net/starcraft2/Resources
        if (currentLoop - u->last_seen_game_loop > 35 || (u->last_seen_game_loop == currentLoop && (u->orders.size() == 0 || u->orders[0].ability_id != ABILITY_ID::HARVEST_GATHER))) {
            // Not a gas harvester anymore
            recentlySeedGasHarvesters.erase(recentlySeedGasHarvesters.begin() + i);
        } else if (u->last_seen_game_loop != currentLoop) {
            numHiddenHarvesters++;
        }
    }

    int foundHarvesters = 0;
    for (const Unit* u : ourUnits) {
        if (isBasicHarvester(u->unit_type)) foundHarvesters += 1;

        if (u->orders.size() > 0 && u->orders[0].ability_id == ABILITY_ID::HARVEST_GATHER && u->orders[0].target_unit_tag != NullTag) {
            auto* targetUnit = agent->Observation()->GetUnit(u->orders[0].target_unit_tag);
            if (targetUnit != nullptr && isVespeneHarvester(targetUnit->unit_type)) {
                if (!contains(recentlySeedGasHarvesters, u)) recentlySeedGasHarvesters.push_back(u);
            }
        }
    }

   return numHiddenHarvesters;
}


BuildOrderTracker::BuildOrderTracker (BuildOrder buildOrder) : buildOrderUnits(buildOrder.size()), buildOrder(buildOrder) {
}

void BuildOrderTracker::setBuildOrder (BuildOrder buildOrder) {
    this->buildOrder = buildOrder;
    buildOrderUnits = vector<const Unit*>(buildOrder.size());
    assert(buildOrder.size() == buildOrderUnits.size());
    cout << "Created as " << buildOrder.size() << " " << buildOrderUnits.size() << endl;
}

void BuildOrderTracker::tweakBuildOrder (std::vector<bool> keepMask, BuildOrder tweakBuildOrder) {
    assert(keepMask.size() == buildOrder.size());
    vector<const Unit*> newUnits;
    BuildOrder newBO;

    for (size_t i = 0; i < buildOrder.size(); i++) {
        if (keepMask[i]) {
            newBO.items.push_back(buildOrder[i]);
            newUnits.push_back(buildOrderUnits[i]);
        }
    }
    
    for (auto item : tweakBuildOrder.items) {
        newBO.items.push_back(item);
        newUnits.push_back(nullptr);
    }

    buildOrder = move(newBO);
    buildOrderUnits = move(newUnits);
}

void BuildOrderTracker::addExistingUnit(const Unit* unit) {
    knownUnits.insert(unit);
}

void BuildOrderTracker::ignoreUnit (UNIT_TYPEID type, int count) {
    ignoreUnits[canonicalize(type)] += count;
}

vector<bool> BuildOrderTracker::update(const vector<const Unit*>& ourUnits) {
    assert(buildOrder.size() == buildOrderUnits.size());
    int loop = agent->Observation()->GetGameLoop();
    for (auto*& u : buildOrderUnits) {
        if (u != nullptr) {
            // If the unit is dead
            // Not sure if e.g. workers that die inside of a refinery are marked as dead properly
            // or if they just disappear. So just to be on the same side we make sure we have seen the unit somewhat recently.
            bool deadOrGone = !u->is_alive || u->last_seen_game_loop - loop > 60;

            // If a structure is destroyed then we should rebuild it to ensure the build order remains valid.
            // However if units die that is normal and we should not try to rebuild them.
            if (deadOrGone && (isStructure(u->unit_type) || u->unit_type == UNIT_TYPEID::ZERG_OVERLORD)) {
                u = nullptr;
            }
        }
    }

    map<UNIT_TYPEID, int> inProgress;
    for (auto* u : ourUnits) {
        if (!knownUnits.count(u)) {
            knownUnits.insert(u);
            auto canonicalType = canonicalize(u->unit_type);

            // New unit
            if (ignoreUnits[canonicalType] > 0) {
                ignoreUnits[canonicalType]--;
                continue;
            }

            for (int i = 0; i < buildOrder.size(); i++) {
                if (buildOrderUnits[i] == nullptr && (buildOrder[i].rawType() == canonicalType)) {
                    buildOrderUnits[i] = u;
                    break;
                }
            }
        }

        for (auto order : u->orders) {
            auto createdUnit = abilityToUnit(order.ability_id);
            if (createdUnit != UNIT_TYPEID::INVALID) {
                if (order.target_unit_tag != NullTag) {
                    auto* unit = bot->Observation()->GetUnit(order.target_unit_tag);
                    // This unit is already existing and is being constructed.
                    // It will already have been associated with the build order item
                    if (unit != nullptr && unit->unit_type == createdUnit) continue;
                }
                inProgress[canonicalize(createdUnit)]++;
            }

            // Only process the first order (this bot should never have more than one anyway)
            break;
        }
    }

    // cout << "In progress ";
    // for (auto p : inProgress) cout << getUnitData(p.first).name << " x" << p.second << ", ";
    // cout << endl;

    
    vector<bool> res(buildOrderUnits.size());
    for (int i = 0; i < buildOrderUnits.size(); i++) {
        res[i] = buildOrderUnits[i] != nullptr;
        if (!buildOrder[i].isUnitType()) {
            auto hasUpgrade = HasUpgrade(buildOrder[i].upgradeID()).Tick();
            res[i] = hasUpgrade == Status::Running || hasUpgrade == Status::Success;
        }

        // No concrete unit may be associated with the item yet, but it may be on its way
        if (!res[i] && inProgress[buildOrder[i].rawType()] > 0) {
            inProgress[buildOrder[i].rawType()]--;
            res[i] = true;
        }
    }
    return res;
}

pair<int, vector<bool>> executeBuildOrder(const vector<const Unit*>& ourUnits, const BuildState& buildOrderStartingState, BuildOrderTracker& tracker, float currentMinerals, SpendingManager& spendingManager, bool& serialize) {
    // Optimize the current build order, but only if we didn't just do an action because then the 'doneActions' list might be inaccurate
    if ((agent->Observation()->GetGameLoop() % 10) == 0 && agent->Observation()->GetGameLoop() - spendingManager.lastActionFrame > 2) {
        BuildState currentState(agent->Observation(), Unit::Alliance::Self, Race::Protoss, BuildResources(agent->Observation()->GetMinerals(), agent->Observation()->GetVespene()), 0);
        optimizeExistingBuildOrder(ourUnits, currentState, tracker, serialize);
        serialize = false;
    }

    auto doneActions = tracker.update(ourUnits);

    // Keep track of how many units have been created/started to be created since the build order was last updated.
    // This will allow us to ensure that we don't do actions multiple times
    /*map<UNIT_TYPEID, int> startingUnitsDelta;
    for (int i = 0; i < ourUnits.size(); i++) {
        // TODO: What about partially constructed buildings which have no worker assigned to it?
        // Terran workers stay with the building while it is being constructed while zerg/protoss workers do not
        if (ourUnits[i]->build_progress < 1 && getUnitData(ourUnits[i]->unit_type).race == Race::Terran) continue;

        startingUnitsDelta[canonicalize(ourUnits[i]->unit_type)]++;
        for (auto order : ourUnits[i]->orders) {
            auto createdUnit = abilityToUnit(order.ability_id);
            if (createdUnit != UNIT_TYPEID::INVALID) {
                startingUnitsDelta[canonicalize(createdUnit)]++;
            }
        }
    }

    for (auto s : buildOrderStartingState.units) startingUnitsDelta[s.type] -= s.units;
    startingUnitsDelta[getHarvesterUnitForRace(buildOrderStartingState.race)] += countMissingVespeneWorkers(ourUnits);*/

    int s = 0;

    int index = 0;
    int currentBuildOrderIndex = 0;

    int totalMinerals = currentMinerals;
    // vector<bool> doneActions(tracker.buildOrder.size());

    int skipCounter = 0;

    for (int i = 0; i < tracker.buildOrder.size(); i++) {
        // Everything requires minerals
        // If we have already reserved all the minerals we have then skip evaluating the rest of the nodes for performance
        if (totalMinerals <= 0) {
            skipCounter++;
            // Note: process a few items even though we are out of minerals
            // This helps to set doneActions accurately and may also help with making probes move to the construction spots before we even get enough minerals.
            if (skipCounter > 2) break;
        }

        if (doneActions[i]) {
            currentBuildOrderIndex = i + 1;
            continue;
        }

        shared_ptr<TreeNode> node = nullptr;
        Cost cost;
        if (tracker.buildOrder[i].isUnitType()) {
            auto b = tracker.buildOrder[i].typeID();
            // Skip the action if it is likely that we have already done it
            /*if (startingUnitsDelta[b] > 0) {
                startingUnitsDelta[b]--;
                currentBuildOrderIndex = i + 1;
                doneActions[i] = true;
                continue;
            }*/

            index++;
            // if (index > 10) break;

            s -= 1;
            if (isVespeneHarvester(b)) {
                node = make_shared<BuildGas>(b, [=](auto) { return s; });
            } else if (isAddon(b)) {
                auto ability = getUnitData(b).ability_id;
                node = make_shared<Addon>(ability, abilityToCasterUnit(ability), [=](auto) { return s; });
            } else if (isTownHall(b)) {
                node = make_shared<Expand>(b, [=](auto) { return s; });
            } else if (isStructure(getUnitData(b))) {
                node = make_shared<Construct>(b, [=](auto) { return s; });
            } else {
                node = make_shared<Build>(b, [=](auto) { return s; }, tracker.buildOrder[i].chronoBoosted);
            }

            cost = CostOfUnit(b);
        } else {
            UpgradeID upgrade = tracker.buildOrder[i].upgradeID();
            /*auto hasAlready = HasUpgrade(upgrade).Tick();
            if (hasAlready == Status::Success || hasAlready == Status::Running) {
                currentBuildOrderIndex = i + 1;
                doneActions[i] = true;
                continue;
            }*/

            index++;
            // if (index > 10) break;

            s -= 1;
            node = make_shared<Research>(upgrade, [=](auto) { return s; });

            cost = CostOfUpgrade(upgrade);
        }

        totalMinerals -= cost.minerals;
        
        // If the action failed, ensure that we reserve the cost for it anyway
        if (node->Tick() == Status::Failure) {
            spendingManager.AddAction(s, cost, []() {}, true);
        }
    }

    return { currentBuildOrderIndex, doneActions };
}

void debugBuildOrderMasked(BuildState startingState, BuildOrder buildOrder, vector<bool> doneItems) {
    BuildState state = startingState;
    stringstream ss;
    BuildOrder maskedBO;
    for (size_t i = 0; i < buildOrder.size(); i++) {
        if (!doneItems[i]) {
            maskedBO.items.push_back(buildOrder[i]);
        } else {
            string name = buildOrder[i].isUnitType() ? getUnitData(buildOrder[i].typeID()).name : UpgradeIDToName(buildOrder[i].upgradeID());
            ss << " :   " << name << "(done)" << endl;
        }
    }
    float timeOffset = ticksToSeconds(agent->Observation()->GetGameLoop());
    // ss << "Time: " << (int)round(buildOrderTime) << endl;
    bool success = state.simulateBuildOrder(maskedBO, [&](int i) {
        string name = maskedBO[i].isUnitType() ? getUnitData(maskedBO[i].typeID()).name : UpgradeIDToName(maskedBO[i].upgradeID());
        float time = state.time + timeOffset;
        int sec = (int)(fmod(time, 60.0f));
        ss << (int)(time / 60.0f) << ":" << (sec < 10 ? "0" : "") << sec << " " << name;
        if (maskedBO.items[i].chronoBoosted) ss << " (chrono)";
        ss << endl;
    });

    if (!success) ss << "FAILED " << "(food: " << state.foodAvailableInFuture() << ")" << endl;

    /*    
    for (int i = 0; i < buildOrder.items.size(); i++) {
        auto b = buildOrder.items[i];
        ss << b.name();
        if (i < currentBuildOrderIndex) ss << " (done)";
        ss << endl;
    }*/
    agent->Debug()->DebugTextOut(ss.str(), Point2D(0.05, 0.05), Colors::Purple);
}

void debugBuildOrder(BuildState startingState, BuildOrder buildOrder, vector<bool> doneItems) {
    BuildState state = startingState;
    stringstream ss;
    // ss << "Time: " << (int)round(buildOrderTime) << endl;
    bool success = state.simulateBuildOrder(buildOrder, [&](int i) {
        string name = buildOrder[i].isUnitType() ? getUnitData(buildOrder[i].typeID()).name : UpgradeIDToName(buildOrder[i].upgradeID());
        int sec = (int)(fmod(state.time, 60.0f));
        ss << (int)(state.time / 60.0f) << ":" << (sec < 10 ? "0" : "") << sec << " " << name;
        if (buildOrder.items[i].chronoBoosted) ss << " (chrono)";
        if (i < doneItems.size() && doneItems[i]) ss << " (done)";
        ss << endl;
    });

    if (!success) ss << "FAILED " << "(food: " << state.foodAvailableInFuture() << ")" << endl;

    /*    
    for (int i = 0; i < buildOrder.items.size(); i++) {
        auto b = buildOrder.items[i];
        ss << b.name();
        if (i < currentBuildOrderIndex) ss << " (done)";
        ss << endl;
    }*/
    if (agent != nullptr) {
        cout << ss.str();
    } else {
        agent->Debug()->DebugTextOut(ss.str(), Point2D(0.05, 0.05), Colors::Purple);
    }
}

void debugBuildOrder(BuildState startingState, BuildOrder buildOrder, vector<float> doneTimes) {
    BuildState state = startingState;
    stringstream ss;
    // ss << "Time: " << (int)round(buildOrderTime) << endl;
    bool success = state.simulateBuildOrder(buildOrder, [&](int i) {
        string name = buildOrder[i].isUnitType() ? getUnitData(buildOrder[i].typeID()).name : UpgradeIDToName(buildOrder[i].upgradeID());
        int sec = (int)(fmod(state.time, 60.0f));
        ss << (int)(state.time / 60.0f) << ":" << (sec < 10 ? "0" : "") << sec << " " << name;
        // if (buildOrder.items[i].chronoBoosted) ss << " (chrono)";
        if (i < doneTimes.size() && doneTimes[i] >= 0) ss << " done: " << (int)(doneTimes[i]/60.0f) << ":" << (int)fmod(doneTimes[i], 60);
        ss << endl;
    });

    /*    
    for (int i = 0; i < buildOrder.items.size(); i++) {
        auto b = buildOrder.items[i];
        ss << b.name();
        if (i < currentBuildOrderIndex) ss << " (done)";
        ss << endl;
    }*/
    agent->Debug()->DebugTextOut(ss.str(), Point2D(0.05, 0.05), Colors::Purple);
}
