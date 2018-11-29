#include "DeductionManager.h"
#include <sstream>
#include "DependencyAnalyzer.h"
#include "utilities/mappings.h"
#include "utilities/predicates.h"
#include "bot.h"
#include "utilities/stdutils.h"

using namespace std;
using namespace sc2;

map<Tag, set<UNIT_TYPEID>> aliasTypes;
Race enemyRace;

void DeductionManager::OnGameStart() {
    const auto& unitTypes = bot.Observation()->GetUnitTypeData();

    expectedObservations = vector<int>(unitTypes.size());

    auto playerInfos = bot.Observation()->GetGameInfo().player_info;
    int ourID = bot.Observation()->GetPlayerID();
    for (auto& p : playerInfos) {
        if (p.player_id != ourID) {
            enemyRace = p.race_requested;
            startingResources = Spending(50, 0);

            // Set the initial units that the players start with
            switch (p.race_requested) {
                case Race::Terran:
                    ExpectObservation(UNIT_TYPEID::TERRAN_COMMANDCENTER, 1);
                    ExpectObservation(UNIT_TYPEID::TERRAN_SCV, 12);
                    // Set the resources that are ignored due to free starting units
                    freeResources = Spending(400 + 12 * 50, 0);
                    break;
                case Race::Protoss:
                    ExpectObservation(UNIT_TYPEID::PROTOSS_NEXUS, 1);
                    ExpectObservation(UNIT_TYPEID::PROTOSS_PROBE, 12);
                    freeResources = Spending(400 + 12 * 50, 0);
                    break;
                case Race::Zerg:
                    ExpectObservation(UNIT_TYPEID::ZERG_HATCHERY, 1);
                    ExpectObservation(UNIT_TYPEID::ZERG_DRONE, 12);
                    ExpectObservation(UNIT_TYPEID::ZERG_OVERLORD, 1);
                    freeResources = Spending(300 + 12 * 50 + 100, 0);
                    break;
                case Race::Random:
                    cout << "Opponent has random race, not sure what starting units they have" << endl;
                    break;
            }
        }
    }
}

void DeductionManager::Observe(vector<const Unit*>& units) {
    const auto& unitTypes = bot.Observation()->GetUnitTypeData();

    for (const Unit* unit : units) {
        Observe(unit);
    }

    int requiredFood = 0;

    cout << endl;
    cout << "Enemy has spent at least " << spending.spentMinerals << " + " << spending.spentGas << endl;
    auto infos = Summary();
    for (int i = 0; i < infos.size(); i++) {
        auto info = infos[i];
        if (info.total > 0) {
            cout << "Has at least " << info.total << " " << unitTypes[i].name << " (" << info.alive << " alive)"
                 << " alias: " << unitTypes[(int)unitTypes[i].unit_alias].name << endl;
        }

        if (info.alive > 0) {
            const UnitTypeData& unitTypeData = unitTypes[i];
            requiredFood += info.alive * unitTypeData.food_required;
            if (unitTypeData.food_provided > 0) {
                UNIT_TYPEID unitType = (UNIT_TYPEID)i;
                // Account for food from command centers and similar
                if (unitType != UNIT_TYPEID::ZERG_OVERLORD && unitType != UNIT_TYPEID::ZERG_OVERSEER && unitType != UNIT_TYPEID::TERRAN_SUPPLYDEPOT && unitType != UNIT_TYPEID::PROTOSS_PYLON) {
                    requiredFood -= unitTypeData.food_provided;
                }
            }
        }
    }

    // The enemy must have a way to get the remaining food, it must be using supply depots/corresponding or new bases, but don't assume that right now.
    UNIT_TYPEID supplyType;
    switch (enemyRace) {
        case Race::Terran:
            supplyType = UNIT_TYPEID::TERRAN_SUPPLYDEPOT;
            break;
        case Race::Protoss:
            supplyType = UNIT_TYPEID::PROTOSS_PYLON;
            break;
        case Race::Zerg:
            supplyType = UNIT_TYPEID::ZERG_OVERLORD;
            break;
        default:
            assert(false);
            break;
    }
    const int foodPerSupplyUnit = 8;
    // Round up expected number of supply depots.
    // Note that we *set* this value every time, so this may become lower if we for example discover a new enemy base (as a command center also provides some food).
    expectedObservations[(int)supplyType] = (requiredFood + (foodPerSupplyUnit - 1)) / foodPerSupplyUnit;
}

vector<UnitTypeInfo> DeductionManager::Summary() {
    const auto& unitTypes = bot.Observation()->GetUnitTypeData();
    spending.spentMinerals = 0;
    spending.spentGas = 0;
    uint32_t gameLoop = bot.Observation()->GetGameLoop();

    vector<UNIT_TYPEID> observedUnitTypes(observedUnitInstances.size());
    vector<int> currentExpectedObservations = expectedObservations;

    for (int i = 0; i < currentExpectedObservations.size(); i++) {
        int toRemove = currentExpectedObservations[i];
        if (toRemove == 0)
            continue;

        auto unitType = (UNIT_TYPEID)i;

        // Go through each unit type this expectation could have been in the past
        // e.g. hive => [hive -> lair -> hatchery -> drone]
        // however since we will do this for every unit type, we must be careful to avoid double counting.
        // For example if we have the expectations
        // [hive: 1, lair: 1, drone: 4]
        // then this should be processed into
        // [hive: 1, lair: 0, drone: 3]
        // as the lair was probably upgraded to a hive, and a single drone built the hatchery which was upgraded to the lair.
        for (UNIT_TYPEID previous : hasBeen(unitType)) {
            // Ignore the type itself (first element in the list)
            if (previous == (UNIT_TYPEID)i)
                continue;

            // Remove expected observations of that type
            int& v = currentExpectedObservations[(int)previous];
            int o = v;
            v = max(0, v - toRemove);
            toRemove = max(0, toRemove - o);
        }
    }

    vector<bool> processedUnits(observedUnitInstances.size());
    // TODO: Does not remove existing drone units (the api does not mark them as dead)
    // TODO: Take infestor::neural parasite into account (mind controlling a unit from another race adds a ton of dependencies that are not actually accurate)
    // TODO: Dark archon can also do mind control?

    // Process units with short upgrade paths first.
    // E.g. if we have both a drone and a hatchery, and we think the enemy has a hive
    // then it is more likely that the hatchery was upgraded to a hive rather than a drone.

    for (int maxUpgradeDistance = 0; maxUpgradeDistance <= 3; maxUpgradeDistance++) {
        for (int i = 0; i < observedUnitInstances.size(); i++) {
            if (processedUnits[i])
                continue;

            const Unit* u = observedUnitInstances[i];

            auto unitType = canonicalize(u->unit_type);
            observedUnitTypes[i] = unitType;
            const UnitTypeData& unitTypeData = unitTypes[(int)unitType];
            bool isVisible = u->last_seen_game_loop == gameLoop;

            bool skip = false;

            if (!isVisible && u->is_alive) {
                // If the unit is not visible then this unit could have been upgraded to something else.
                // If it is dead we presumably know the last possible type of it, so it couldn't have been upgraded.
                // Check all units it could have been in the future:
                // e.g: command center => [orbital command, planetary fortress]
                // or
                // hatchery => [lair, hive]
                // Loop through them in reverse order (i.e most expensive one first)
                for (UNIT_TYPEID future : reverse(canBecome(unitType))) {
                    int& v = currentExpectedObservations[(int)future];
                    if (v > 0) {
                        // Check what the upgrade depth is.
                        // e.g. drone -> hatchery = 1
                        // drone -> (hatchery -> lair) -> hive = 3
                        auto upgradePath = hasBeen(future);
                        auto depth = find(upgradePath.begin(), upgradePath.end(), unitType) - upgradePath.begin();
                        assert(depth < upgradePath.size());
                        assert(depth <= 3);

                        if (depth > maxUpgradeDistance) {
                            // Too high depth, try again in the next iteration
                            skip = true;
                            break;
                        }

                        // Ok, assume this unit has been upgraded to #future
                        cout << "Assuming " << UnitTypeToName(unitType) << " has been upgraded to " << UnitTypeToName(future) << endl;

                        unitType = future;

                        break;
                    }
                }
            }

            if (skip)
                continue;

            // Check all types that this unit has been in the past (or is right now)
            // e.g hive => [hive -> lair -> hatchery -> drone]
            // or orbital command => [orbital command -> command center]
            for (UNIT_TYPEID previous : hasBeen(unitType)) {
                // Remove an expected observation of that type
                int& v = currentExpectedObservations[(int)previous];
                if (v > 0) {
                    v--;
                }
            }

            observedUnitTypes[i] = unitType;
            processedUnits[i] = true;
        }
    }

    vector<UnitTypeInfo> infos(unitTypes.size());
    for (int i = 0; i < observedUnitInstances.size(); i++) {
        const Unit* u = observedUnitInstances[i];
        // Note: already canonicalized
        auto unitType = observedUnitTypes[i];
        auto& info = infos[(int)unitType];
        info.total += 1;
        info.alive += u->is_alive;
        info.dead += !u->is_alive;

        const UnitTypeData& unitTypeData = unitTypes[(int)unitType];
        spending.spentMinerals += unitTypeData.mineral_cost;
        spending.spentGas += unitTypeData.vespene_cost;
    }

    for (int i = 0; i < currentExpectedObservations.size(); i++) {
        int expected = currentExpectedObservations[i];
        infos[i].total += expected;
        infos[i].alive += expected;

        const UnitTypeData& unitTypeData = unitTypes[i];
        spending.spentMinerals += expected * unitTypeData.mineral_cost;
        spending.spentGas += expected * unitTypeData.vespene_cost;
    }

    spending.spentMinerals -= freeResources.spentMinerals;
    spending.spentGas -= freeResources.spentGas;

    return infos;
}

string DisplayTypeToString(Unit::DisplayType x) {
    switch (x) {
        case Unit::DisplayType::Hidden:
            return "Hidden";
        case Unit::DisplayType::Snapshot:
            return "Snapshot";
        case Unit::DisplayType::Visible:
            return "Visible";
    }
}

/** Observe a particular unit instance */
void DeductionManager::Observe(const Unit* unit) {
    aliasTypes[unit->tag].insert(unit->unit_type);

    // Snapshots are ignored, we will have seen the real thing before
    if (unit->display_type == Unit::DisplayType::Snapshot) {
        return;
    }

    auto canonicalUnitType = canonicalize(unit->unit_type);
    const auto& unitTypes = bot.Observation()->GetUnitTypeData();
    const UnitTypeData& unitTypeData = unitTypes[(int)canonicalUnitType];

    auto alreadyObservedType = observedUnits.find(unit->tag);

    if (alreadyObservedType == observedUnits.end()) {
        observedUnitInstances.push_back(unit);
    }

    observedUnits[unit->tag] = canonicalUnitType;

    Observe(canonicalUnitType);
}

void DeductionManager::ExpectObservation(UNIT_TYPEID unitType, int count) {
    int& v = expectedObservations[(int)unitType];
    v = max(v, count);
    if (count > 0)
        observedUnitTypes.insert(unitType);
}

/** Observe the unit type, can be used to note that we know that the player has had at least one of the specified type */
void DeductionManager::Observe(UNIT_TYPEID unitType) {
    unitType = canonicalize(unitType);

    const auto& unitTypes = bot.Observation()->GetUnitTypeData();
    const UnitTypeData& unitTypeData = unitTypes[(int)unitType];

    if (observedUnitTypes.find(unitType) != observedUnitTypes.end()) {
        // Already seen this one
        return;
    }

    vector<UNIT_TYPEID> earlierBuildingUpgrades = hasBeen(unitType);

    observedUnitTypes.insert(unitType);
    ExpectObservation(unitType, 1);

    cout << "Observed type " << UnitTypeToName(unitType) << endl;

    for (auto u : bot.dependencyAnalyzer.allUnitDependencies[(int)unitType]) {
        // Observe dependencies (unless they are just earlier upgrades of the same building type)
        if (!contains(earlierBuildingUpgrades, u)) {
            // Observe the unit type
            Observe(u);
        }
    }
}

// Keep track of all enemy units
// Keep track of all (type, count) of unit types that we know the enemy must have
// Merge by for all existing units (alive or dead):
// If visible: reduce all (type, count) where type is a unit type that this unit could have been before (e.g. command center before orbital command)
// If not visible: do same as visible *and* check if there is a (type, count) where type is a unit type that this unit could be in the future (e.g orbital command after command center) and if so: assume this unit is of that type (if multiple, use the one with the highest tech level).