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

bool isBasicHarvester(UNIT_TYPEID type) {
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
        UNIT_TYPEID unit = abilityToUnit(ability);
        return isBasicHarvester(unit) || isStructure(unit) || supply(unit) > 0;
    }

    void apply(BuildState& state) {
        switch(type) {
            case FinishedUnit: {
                UNIT_TYPEID unit = abilityToUnit(ability);
                UNIT_TYPEID caster = abilityToCasterUnit(ability);
                state.makeUnitsBusy(caster, -1);
                state.addUnits(unit, 1);
                
                // First element is the unit itself
                // second element is the one it was created from
                // third element is the one that one was created from etc.
                // E.g. hasBeen[hatchery][1] = drone
                //      hasBeen[lair][1] = hatchery
                auto& hasBeenUnits = hasBeen(unit);
                if (hasBeenUnits.size() > 1) {
                    state.addUnits(hasBeenUnits[1], -1);
                }
                break;
            }
            case SpawnLarva: {
                state.addUnits(UNIT_TYPEID::ZERG_LARVA, 3);
                break;
            }
            case MuleTimeout: {
                state.addUnits(UNIT_TYPEID::TERRAN_MULE, -1);
                break;
            }
        }
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

    void simulateMining(pair<float, float> miningSpeed, float dt) {
        minerals += miningSpeed.first * dt;
        vespene += miningSpeed.second * dt;
    }
};

struct BuildOptimizerSearch {
    priority_queue<shared_ptr<BuildState>> que;

    void addState(shared_ptr<BuildState> state) {
        que.insert(make_pair(upperBound(state), state));
    }

    void search () {
        float lowestUpperBound = 1000000;
        shared_ptr<BuildState> bestResult = nullptr;

        while(!que.empty()) {
            auto state = que.front();
            que.pop();

            if (lowerBound(state) > lowestUpperBound) {
                // This state cannot possibly be optimal, so skip it
                continue;
            }

            float ub = upperBound(state);
            if (ub < lowestUpperBound) {
                lowestUpperBound = ub;
                bestResult = state;
            }

            // If it's not done, add more actions
            if (state.time < ub) {
                state.doActions(*this);
            }
        }
    }
}
 
struct BuildState {
    float time;
    BuildAction parentAction;
    shared_ptr<BuildState> parent;

    vector<BuildUnitInfo> units;
    vector<BuildEvent> events;
    BuildResources resources;

    void makeUnitsBusy(UNIT_TYPEID type, int delta) {
        for (auto& u : state.units) {
            if (u.type == type) {
                u.busyUnits += delta;
                assert(u.busyUnits <= u.units);
                return;
            }
        }
        assert(false);
    }

    void addUnits(UNIT_TYPEID type, int delta) {
        if (delta < 0) {
            for (auto& u : state.units) {
                if (u.type == type) {
                    assert(u.units > 0);
                    u.units -= delta;
                    assert(u.busyUnits <= u.units);
                    return;
                }
            }
            assert(false);
        } else if (delta > 0) {
            for (auto& u : state.units) {
                if (u.type == type) {
                    u.units += delta;
                    return;
                }
            }

            BuildUnitInfo info;
            info.units = delta;
            info.busyUnits = 0;
            info.type = type;
            state.units.push_back(info);
        }
    }

    pair<float, float> miningSpeed() const {
        int harvesters = 0;
        int mules = 0;
        int bases = 0;
        int geysers = 0;
        for (auto& unit : units) {
            // TODO: Normalize type?
            if (isBasicHarvester(unit.type)) {
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

    // All actions up to and including the end time will have been completed
    void simulate(float endTime) {
        auto currentMiningSpeed = miningSpeed();
        for (auto& ev : events) {
            if (ev.time > endTime) {
                events.remove(...);
                return;
            }
            float dt = ev.time - time;
            resources.simulateMining(currentMiningSpeed, dt);
            time = ev.time;

            ev.apply(*this);

            // TODO:
            //if (ev.impactsEconomy()) {
            currentMiningSpeed = miningSpeed();
        }
        events.clear();
    }

    void doActions(BuildOptimizerSearch& search) const {
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
                    // Edge = can do in this state, but couldn't in the parent state (excluding minerals/gas)
                    if (unitData.mineral_cost >= resources.minerals || unitData.vespene_cost >= resources.vespene) {
                        float eventTime = time + timeToGetResources(currentMiningSpeed, unitData.mineral_cost, unitData.vespene_cost);
                        if (eventTime < nextSignificantEvent) {
                            // Simulate up to this point
                            BuildState newState = *this;
                            newState.simulate(eventTime);
                            newState.resources.minerals -= unitData.mineral_cost;
                            newState.resources.vespene -= unitData.vespene_cost;
                            newState.addEvent(BuildEvent(BuildEventType::FinishedUnit, newState.time + unitData.build_time, ability));
                            search.addState(newState);
                        }
                    }
                }
            }
        }
    }
};

// The lower time this state can possibly lead to reaching the target
float lowerBound(const BuildState& state, const BuildTarget& target) {
    const auto& unitTypes = agent.Observation()->GetUnitTypeData();

    // Build all units in parallel
    // don't account for economy (yet)
    vector<pair<UNIT_TYPEID, int>> remainingUnits = target.units;
    for (auto& u : state.units) {
        for (auto& u2 : remainingUnits) {
            if (u2.first == u.type) {
                u2 = make_pair(u2.first, max(0, u2.second - u.units));
            }
        }
    }

    float maxRelevantEventTime = 0;
    for (auto& ev : state.events) {
        if (ev.type == FinishedUnit) {
            UNIT_TYPEID unit = abilityToUnit(ev.ability);
            for (auto& u2 : remainingUnits) {
                if (u2.first == unit && u2.second > 0) {
                    maxRelevantEventTime = max(maxRelevantEventTime, ev.time);
                    u2 = make_pair(u2.first, u2.second - 1);
                }
            }
        }
    }

    float requiredMinerals = 0;
    float requiredVespene = 0;
    float maxBuildTime = 0;
    for (auto& u : remainingUnits) {
        auto& unitData = unitTypes[(int)u.first];
        maxBuildTime = max(maxBuildTime, unitData.build_time);
    }
    return max(state.time + maxBuildTime, maxRelevantEventTime);
}

// It is guaranteed that the target can be reached at least at this time from this state
float upperBound(const BuildState& state, const BuildTarget& target) {
    // 3. Get the required minerals/vespene
    // 1. Get the required buildings
    // 2. Get the required supply
    // 3. Get the required units
    // Maybe formulate as simulating a specific build order
    // [buildings, supply, units], might be too slow though
    
    // Go through unit dependencies, find time to get at least one of that building
    // Build all units sequentially (or using the available buildings)
    // TODO: Supply
}

void BuildOptimizer::calculate_build_order(vector<pair<UNIT_TYPEID, int>> start, vector<pair<UNIT_TYPEID, int>> target) {
}
