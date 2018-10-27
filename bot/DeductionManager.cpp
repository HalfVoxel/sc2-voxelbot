#include "DeductionManager.h"
#include <sstream>
#include "DependencyAnalyzer.h"
#include "Mappings.h"
#include "Predicates.h"
#include "bot.h"
#include "stdutils.h"

using namespace std;
using namespace sc2;

map<Tag, set<UNIT_TYPEID>> aliasTypes;

void DeductionManager::OnGameStart() {
    const auto& unitTypes = bot.Observation()->GetUnitTypeData();

    expectedObservations = vector<int>(unitTypes.size());

    auto playerInfos = bot.Observation()->GetGameInfo().player_info;
    int ourID = bot.Observation()->GetPlayerID();
    for (auto& p : playerInfos) {
        if (p.player_id != ourID) {
            // Players start with 50 minerals
            spending.spentMinerals = -50;
            // Set the initial units that the players start with
            switch (p.race_requested) {
                case Race::Terran:
                    ExpectObservation(UNIT_TYPEID::TERRAN_COMMANDCENTER, 1);
                    ExpectObservation(UNIT_TYPEID::TERRAN_SCV, 12);
                    break;
                case Race::Protoss:
                    ExpectObservation(UNIT_TYPEID::PROTOSS_NEXUS, 1);
                    ExpectObservation(UNIT_TYPEID::PROTOSS_PROBE, 12);
                    break;
                case Race::Zerg:
                    ExpectObservation(UNIT_TYPEID::ZERG_HATCHERY, 1);
                    ExpectObservation(UNIT_TYPEID::ZERG_DRONE, 12);
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

    int requiredFood = 0;
    for (const Unit* unit : units) {
        Observe(unit);

        const UnitTypeData& unitType = unitTypes[unit->unit_type];
        requiredFood += unitType.food_required;
    }

    cout << endl;
    cout << "Enemy has spent at least " << spending.spentMinerals << " + " << spending.spentGas << endl;
    auto infos = Summary();
    for (int i = 0; i < infos.size(); i++) {
        auto info = infos[i];
        if (info.total > 0) {
            cout << "Has at least " << info.total << " " << unitTypes[i].name << " (" << info.alive << " alive)"
                 << " alias: " << unitTypes[(int)unitTypes[i].unit_alias].name << endl;
        }
    }
}

UNIT_TYPEID canonicalize(UNIT_TYPEID unitType) {
    const auto& unitTypes = bot.Observation()->GetUnitTypeData();
    const UnitTypeData& unitTypeData = unitTypes[(int)unitType];

    // Use canonical representation (e.g SUPPLY_DEPOT instead of SUPPLY_DEPOT_LOWERED)
    if (unitTypeData.unit_alias != UNIT_TYPEID::INVALID) {
        return unitTypeData.unit_alias;
    }
    return unitType;
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

    // TODO: Does not remove existing drone units (the api does not mark them as dead)
    for (int i = 0; i < observedUnitInstances.size(); i++) {
        const Unit* u = observedUnitInstances[i];

        auto unitType = canonicalize(u->unit_type);
        observedUnitTypes[i] = unitType;
        const UnitTypeData& unitTypeData = unitTypes[(int)unitType];
        bool isVisible = u->last_seen_game_loop == gameLoop;

        if (!isVisible && !u->is_alive) {
            // If the unit is not visible then this unit could have been upgraded to something else.
            // Check all units it could have been in the future:
            // e.g: command center => [orbital command, planetary fortress]
            // Loop through them in reverse order (i.e most expensive one first)
            for (UNIT_TYPEID future : reverse(hasBeen(unitType))) {
                // Remove an expected observation of that type
                int& v = currentExpectedObservations[(int)future];
                if (v > 0) {
                    // Ok, assume this unit has been upgraded to #future
                    unitType = future;
                }
            }
        }

        // Check all types that this unit has been in the past (or is right now)
        // e.g hive => [hive -> lair -> hatchery -> drone]
        // or orbital command => [orbital command -> command center]

        // Handle building upgrades in a reasonable way.
        // If we expect that the enemy has say a command center
        // and then, say due to a scan effect, we infer that the enemy has an orbital command,
        for (UNIT_TYPEID previous : hasBeen(unitType)) {
            // Remove an expected observation of that type
            int& v = currentExpectedObservations[(int)previous];
            if (v > 0) {
                v--;
            }
        }

        observedUnitTypes[i] = unitType;
    }

    vector<UnitTypeInfo> infos(unitTypes.size());
    for (int i = 0; i < observedUnitInstances.size(); i++) {
        const Unit* u = observedUnitInstances[i];
        auto unitType = canonicalize(u->unit_type);
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