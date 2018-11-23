#include "BuildOptimizer.h"
#include <algorithm>
#include <map>
#include <vector>
#include "Mappings.h"
#include "Bot.h"
#include <limits>

using namespace std;
using namespace sc2;

void BuildOptimizer::init(const ObservationInterface* observation) {
}

bool isHarvester(UNIT_TYPEID type) {
    switch(type) {
        case UNIT_TYPEID::TERRAN_SCV: return true;
        case UNIT_TYPEID::ZERG_DRONE: return true;
        case UNIT_TYPEID::PROTOSS_PROBE: return true;
        default: return false;
    }
}

bool isTownHall(UNIT_TYPEID type) {
    switch(type) {
        case UNIT_TYPEID::ZERG_HATCHERY: return true;
        case UNIT_TYPEID::ZERG_LAIR: return true;
        case UNIT_TYPEID::ZERG_HIVE : return true;
        case UNIT_TYPEID::TERRAN_COMMANDCENTER: return true;
        case UNIT_TYPEID::TERRAN_ORBITALCOMMAND: return true;
        case UNIT_TYPEID::TERRAN_ORBITALCOMMANDFLYING: return true;
        case UNIT_TYPEID::TERRAN_PLANETARYFORTRESS: return true;
        case UNIT_TYPEID::PROTOSS_NEXUS: return true;
        default: return false;
    }
}

bool isVespeneHarvester(UNIT_TYPEID type) {
    switch(type) {
        case UNIT_TYPEID::TERRAN_REFINERY: return true;
        case UNIT_TYPEID::ZERG_EXTRACTOR: return true;
        case UNIT_TYPEID::PROTOSS_ASSIMILATOR : return true;
        default: return false;
    }
}

// Events happen in case of
// 1: units finished (all actions reevaluated)
// 2: resources aquired (minerals/vespene) in order to perform an action (only that action evaluated, might lead to a new event for another action)
// 3: energy aquired in order to perform an action (e.g. mule/spawn larva)

// Pick state with lowest lowerBound? (or maybe state with lowest upper bound)
// Evaluate all possible actions on it
//    try to build all possible buildings (maybe only mineral/vespene edge triggered to keep branching factor down. So a building is only considered to be constructed right when it can first be built, have we ignored it once we will not try to build it again until we no longer have enough resources)
//    try to use all abilities (also edge triggered)
//    try to build all units (edge triggered)
// Essentially
// Loop through all actions
//    only consider actions that we cannot do right now (unless this is a newly constructed building/unit?). I.e. we haven't previously skipped it.
//    calculate the time when it can be done based on current mining rate
//    clone the state and assume it is done at that time

// Model minerals, vespene, larva, and energy as a resource
enum BuildEventType {
    FinishedUnit,
    SpawnLarva,
    MuleTimeout,
};

struct BuildEvent {
    BuildEventType type;
    ABILITY_ID ability;
    float time;

    bool impactsEconomy() const {

    }
};

struct BuildUnitInfo {
    UNIT_TYPEID type;
    int units;
    // E.g. constructing a building, training a unit, etc.
    int busyUnits;

    int availableUnits () const {
        return units - busyUnits;
    }
};

enum BuildActionType {
    CastAbility,
    // DistributeHarvesters?
};

struct BuildAction {
    BuildActionType type;
    ABILITY_ID ability;
};

struct BuildResources {
    float minerals;
    float vespene;
};

struct BuildState {
    float time;
    BuildAction parentAction;
    shared_ptr<BuildState> parent;

    vector<BuildUnitInfo> units;
    vector<BuildEvent> events;
    BuildResources resources;

    pair<float, float> miningSpeed() const {
        int harvesters = 0;
        int mules = 0;
        int bases = 0;
        int geysers = 0;
        for (auto& unit : units) {
            // TODO: Normalize type?
            if (isHarvester(unit.type)) {
                harvesters += unit.availableUnits();
            }

            if (unit.type == UNIT_TYPEID::TERRAN_MULE) {
                mules += unit.availableUnits();
            }

            if (isTownHall(unit.type)) {
                bases += unit.units;
            }

            if (isVespeneHarvester(unit.type)) {
                geysers += unit.units;
            }
        }

        int vespeneMining = min(harvesters/2, geysers*3);
        int mineralMining = harvesters - vespeneMining;

        // Maximum effective harvesters (todo: account for more things)
        mineralMining = min(mineralMining, bases * 16);

        // First 2 harvesters per mineral field yield more minerals than the 3rd one.
        int highYieldHarvesters = min(bases * 8 * 2, mineralMining);
        int lowYieldHarvesters = mineralMining - highYieldHarvesters;
        
        // TODO: Check units here!
        const float FasterSpeedMultiplier = 1.4f;
        const float LowYieldMineralsPerMinute = 22 / FasterSpeedMultiplier;
        const float HighYieldMineralsPerMinute = 40 / FasterSpeedMultiplier;
        const float VespenePerMinute = 38 / FasterSpeedMultiplier;

        float mineralsPerSecond = (lowYieldHarvesters * LowYieldMineralsPerMinute + highYieldHarvesters * HighYieldMineralsPerMinute) / 60.0f;
        float vespenePerSecond = vespeneMining * VespenePerMinute;
        return make_pair(mineralsPerSecond, vespenePerSecond);
    }

    float timeToGetResources (pair<float, float> miningSpeed, float mineralCost, float vespeneCost) const {
        mineralCost -= resources.minerals;
        vespeneCost -= resources.vespene;
        float time = 0;
        if (mineralCost > 0) {
            if (miningSpeed.first == 0) return numeric_limits<float>::infinity();
            time = mineralCost / miningSpeed.first;
        }
        if (vespeneCost > 0) {
            if (miningSpeed.second == 0) return numeric_limits<float>::infinity();
            time = max(time, vespeneCost / miningSpeed.second);
        }
        return time;
    }

    void doActions() const {
        const auto& abilities = agent.Observation()->GetAbilityData();
        const auto& unitTypes = agent.Observation()->GetUnitTypeData();

        float nextSignificantEvent = 10000000;
        for (auto& ev : events) {
            if (ev.impactsEconomy()) nextSignificantEvent = min(nextSignificantEvent, ev.time);
        }

        auto currentMiningSpeed = miningSpeed();

        for (auto& unit : units) {
            for (auto ability : unitAbilities(unit.type)) {
                auto createdUnit = abilityToUnit(ability);
                if (createdUnit != UNIT_TYPEID::INVALID) {
                    auto& abilityData = abilities[(int)ability];
                    auto& unitData = unitTypes[(int)createdUnit];

                    // TODO: Needs to be supply edge triggered as well
                    if (unitData.mineral_cost >= resources.minerals || unitData.vespene_cost >= resources.vespene) {
                        float eventTime = time + timeToGetResources(currentMiningSpeed, unitData.mineral_cost, unitData.vespene_cost);
                    }
                }
            }
        }
    }
};

float lowerBound();
float upperBound();

void BuildOptimizer::calculate_build_order(vector<pair<UNIT_TYPEID, int>> start, vector<pair<UNIT_TYPEID, int>> target) {
}
