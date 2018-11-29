#if FALSE
#include "BuildOptimizer.h"
#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <vector>
#include "Bot.h"
#include "utilities/stdutils.h"
#include "utilities/mappings.h"
#include "utilities/predicates.h"

using namespace std;
using namespace sc2;

set<int> hashes;

bool isBasicHarvester(UNIT_TYPEID type) {
    switch (type) {
        case UNIT_TYPEID::TERRAN_SCV:
            return true;
        case UNIT_TYPEID::ZERG_DRONE:
            return true;
        case UNIT_TYPEID::PROTOSS_PROBE:
            return true;
        default:
            return false;
    }
}

bool isTownHall(UNIT_TYPEID type) {
    switch (type) {
        case UNIT_TYPEID::ZERG_HATCHERY:
            return true;
        case UNIT_TYPEID::ZERG_LAIR:
            return true;
        case UNIT_TYPEID::ZERG_HIVE:
            return true;
        case UNIT_TYPEID::TERRAN_COMMANDCENTER:
            return true;
        case UNIT_TYPEID::TERRAN_ORBITALCOMMAND:
            return true;
        case UNIT_TYPEID::TERRAN_ORBITALCOMMANDFLYING:
            return true;
        case UNIT_TYPEID::TERRAN_PLANETARYFORTRESS:
            return true;
        case UNIT_TYPEID::PROTOSS_NEXUS:
            return true;
        default:
            return false;
    }
}

bool isVespeneHarvester(UNIT_TYPEID type) {
    switch (type) {
        case UNIT_TYPEID::TERRAN_REFINERY:
            return true;
        case UNIT_TYPEID::ZERG_EXTRACTOR:
            return true;
        case UNIT_TYPEID::PROTOSS_ASSIMILATOR:
            return true;
        default:
            return false;
    }
}

UNIT_TYPEID supplyUnit(Race race) {
    switch (race) {
        case Race::Protoss:
            return UNIT_TYPEID::PROTOSS_PYLON;
        case Race::Terran:
            return UNIT_TYPEID::TERRAN_SUPPLYDEPOT;
        case Race::Zerg:
            return UNIT_TYPEID::ZERG_OVERLORD;
        default:
            assert(false);
            return UNIT_TYPEID::INVALID;
    }
}

/** Times in the SC2 API are often defined in ticks, instead of seconds.
 * This method assumes the 'Faster' game speed.
 */
float ticksToSeconds (float ticks) {
    return ticks / 22.4f;
}

bool isStructure(UNIT_TYPEID type) {
    // TODO: Cache in mappings?
    return isStructure(getUnitData(type));
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

struct BuildState;

struct BuildEvent {
    BuildEventType type;
    ABILITY_ID ability;
    UNIT_TYPEID caster;
    float time;

    BuildEvent(BuildEventType type, float time, UNIT_TYPEID caster, ABILITY_ID ability)
        : type(type), time(time), caster(caster), ability(ability) {}
    
    bool impactsEconomy() const;
    void apply(BuildState& state);

    bool operator< (const BuildEvent& other) const {
        return time < other.time;
    }
};

struct BuildTarget {
    Race race;
    vector<pair<UNIT_TYPEID, int>> units;
};

float upperBound(const BuildState& state, const BuildTarget& target);
float lowerBound(const BuildState& state, const BuildTarget& target);

struct BuildUnitInfo {
    UNIT_TYPEID type;
    int units;
    // E.g. constructing a building, training a unit, etc.
    int busyUnits;

    BuildUnitInfo()
        : type(UNIT_TYPEID::INVALID), units(0), busyUnits(0) {}
    BuildUnitInfo(UNIT_TYPEID type, int units)
        : type(type), units(units), busyUnits(0) {}

    int availableUnits() const {
        return units - busyUnits;
    }
};

enum BuildActionType {
    CastAbility,
    Wait,
    // DistributeHarvesters?
};

struct BuildAction {
    BuildActionType type;
    ABILITY_ID ability;
};

struct BuildResources {
    float minerals;
    float vespene;

    BuildResources (float minerals, float vespene) : minerals(minerals), vespene(vespene) {}

    void simulateMining(pair<float, float> miningSpeed, float dt) {
        minerals += miningSpeed.first * dt;
        vespene += miningSpeed.second * dt;
    }
};

struct BuildOptimizerSearch {
    priority_queue<pair<float, shared_ptr<BuildState>>, vector<pair<float, shared_ptr<BuildState>>>, std::greater<pair<float, shared_ptr<BuildState>>>> que;
    BuildTarget& target;

    BuildOptimizerSearch(BuildTarget& target)
        : target(target) {
    }

    void addState(shared_ptr<BuildState> state) {
        que.push(make_pair(lowerBound(*state, target), state));
    }

    shared_ptr<BuildState> search();
};

struct BuildState : std::enable_shared_from_this<BuildState> {
    float time;
    BuildAction parentAction;
    shared_ptr<const BuildState> parent;

    vector<BuildUnitInfo> units;
    mutable vector<pair<UNIT_TYPEID, ABILITY_ID>> possibleActions;
    vector<BuildEvent> events;
    BuildResources resources;

    BuildState () : time(0), parentAction(), parent(nullptr), units(), events(), resources(0, 0) {}

    void makeUnitsBusy(UNIT_TYPEID type, int delta) {
        for (auto& u : units) {
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
            for (auto& u : units) {
                if (u.type == type) {
                    assert(u.units > 0);
                    u.units -= delta;
                    assert(u.busyUnits <= u.units);
                    return;
                }
            }
            assert(false);
        } else if (delta > 0) {
            for (auto& u : units) {
                if (u.type == type) {
                    u.units += delta;
                    return;
                }
            }

            BuildUnitInfo info;
            info.units = delta;
            info.busyUnits = 0;
            info.type = type;
            units.push_back(info);
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

        int vespeneMining = min(harvesters / 2, geysers * 3);
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
        const float MinutesPerSecond = 1/60.0f;
        float mineralsPerSecond = (lowYieldHarvesters * LowYieldMineralsPerMinute + highYieldHarvesters * HighYieldMineralsPerMinute) * MinutesPerSecond;
        float vespenePerSecond = vespeneMining * VespenePerMinute * MinutesPerSecond;
        return make_pair(mineralsPerSecond, vespenePerSecond);
    }

    float timeToGetResources(pair<float, float> miningSpeed, float mineralCost, float vespeneCost) const {
        mineralCost -= resources.minerals;
        vespeneCost -= resources.vespene;
        float time = 0;
        if (mineralCost > 0) {
            if (miningSpeed.first == 0)
                return numeric_limits<float>::infinity();
            time = mineralCost / miningSpeed.first;
        }
        if (vespeneCost > 0) {
            if (miningSpeed.second == 0)
                return numeric_limits<float>::infinity();
            time = max(time, vespeneCost / miningSpeed.second);
        }
        return time;
    }

    void addEvent(BuildEvent event) {
        // TODO: Insertion sort or something
        events.push_back(event);
        sort(events.begin(), events.end());
        for (auto& ev : events) {
            cout << ev.time << " ";
        }
        cout << endl;
    }

    // All actions up to and including the end time will have been completed
    void simulate(float endTime) {
        auto currentMiningSpeed = miningSpeed();
        int eventIndex;
        for (eventIndex = 0; eventIndex < events.size(); eventIndex++) {
            auto& ev = events[eventIndex];
            if (ev.time > endTime) {
                break;
            }
            float dt = ev.time - time;
            resources.simulateMining(currentMiningSpeed, dt);
            time = ev.time;

            ev.apply(*this);

            // TODO:
            //if (ev.impactsEconomy()) {
            currentMiningSpeed = miningSpeed();
        }

        events.erase(events.begin(), events.begin()+eventIndex);
        
        {
            float dt = endTime - time;
            resources.simulateMining(currentMiningSpeed, dt);
            time = endTime;
        }
    }

    bool simulateBuildOrder(vector<UNIT_TYPEID> buildOrder) {
        for (auto unitType : buildOrder) {
            while (true) {
                float nextSignificantEvent = 10000000;
                for (auto& ev : events) {
                    if (ev.impactsEconomy()) {
                        nextSignificantEvent = ev.time;
                        break;
                    }
                }

                auto& unitData = getUnitData(unitType);
                auto currentMiningSpeed = miningSpeed();
                float eventTime = time + timeToGetResources(currentMiningSpeed, unitData.mineral_cost, unitData.vespene_cost);
                if (eventTime > nextSignificantEvent) {
                    simulate(nextSignificantEvent);
                    continue;
                }

                // TODO: Need to handle multiple casters case (e.g. need to be able to make SCVs from planetary fortress)
                ABILITY_ID ability = unitData.ability_id;
                UNIT_TYPEID caster = abilityToCasterUnit(ability)[0];
                BuildUnitInfo* casterUnit = nullptr;
                for (auto& casterCandidate : units) {
                    if (casterCandidate.type == caster && casterCandidate.availableUnits() > 0) {
                        casterUnit = &casterCandidate;
                    }
                }

                if (casterUnit == nullptr) {
                    if (events.empty())
                        return false;
                    simulate(events[0].time);
                    continue;
                }

                resources.minerals -= unitData.mineral_cost;
                resources.vespene -= unitData.vespene_cost;
                makeUnitsBusy(caster, 1);
                addEvent(BuildEvent(BuildEventType::FinishedUnit, time + ticksToSeconds(unitData.build_time), casterUnit->type, ability));
            }
        }

        return true;
    }

    int foodAvailable() const {
        int totalSupply = 0;
        for (auto& unit : units) {
            auto& data = getUnitData(unit.type);
            totalSupply += (data.food_provided - data.food_required) * unit.units;
        }
        // Units in construction use food, but they don't provide food (yet)
        for (auto& ev : events) {
            auto& data = getUnitData(abilityToUnit(ev.ability));
            totalSupply -= data.food_required;
        }

        assert(totalSupply >= 0);
        return totalSupply;
    }

    bool hasEquivalentTech(UNIT_TYPEID type) const {
        for (auto& unit : units) {
            auto& unitData = getUnitData(unit.type);
            if (unit.type == type) {
                return true;
            }
            for (auto t : unitData.tech_alias) if (t == type) return true;
        }
        return false;
    }

    void doActions(BuildOptimizerSearch& search) const;
};

vector<shared_ptr<const BuildState>> traceStates(shared_ptr<const BuildState> state) {
    vector<shared_ptr<const BuildState>> states;
    while (state != nullptr) {
        states.push_back(state);
        state = state->parent;
    }
    reverse(states.begin(), states.end());
    return states;
}

vector<UNIT_TYPEID> extractBuildOrder(vector<shared_ptr<const BuildState>> states) {
    vector<UNIT_TYPEID> buildOrder;
    for (int i = 1; i < states.size(); i++) {
        auto& action = states[i]->parentAction;
        if (action.type == BuildActionType::CastAbility) {
            auto builtUnit = abilityToUnit(action.ability);
            buildOrder.push_back(builtUnit);
        }
    }
    return buildOrder;
}

void logBuildOrder(vector<UNIT_TYPEID> buildOrder) {
    cout << "Build order with " << buildOrder.size() << " steps" << endl;
    for (int i = 0; i < buildOrder.size(); i++) {
        cout << i << " " << UnitTypeToName(buildOrder[i]) << endl;
    }
}

void logDetailedBuildOrder(vector<shared_ptr<const BuildState>> states) {
    cout << "Build order with " << (states.size()-1) << " steps" << endl;
    cout << "Start time " << states[0]->time << endl;
    for (int i = 1; i < states.size(); i++) {
        auto& action = states[i]->parentAction;
        if (action.type == BuildActionType::CastAbility) {
            auto builtUnit = abilityToUnit(action.ability);
            cout << i << " " << UnitTypeToName(builtUnit) << " at " << states[i]->time << endl;
        } else if (action.type == BuildActionType::Wait) {
            cout << i << " wait until " << states[i]->time << endl;
        }
    }
}

vector<ABILITY_ID> blacklistedAbilities = {
    ABILITY_ID::MORPH_SUPPLYDEPOT_LOWER,
    ABILITY_ID::MORPH_SUPPLYDEPOT_RAISE,
};

void BuildState::doActions(BuildOptimizerSearch& search) const {
    int hash = 0;
    for (auto& u : units) {
        hash = hash ^ (51237 * (((int)u.type * 31) ^ (u.units * 7) ^ (u.busyUnits * 95237)));
    }
    hash *= 31;
    // for (auto& u : events) {
    //     hash ^= ((int)u.caster * 31) ^ ((int)u.ability * 523);
    // }
    hashes.insert(hash);
    cout << "Total hashes " << hashes.size() << endl;

    possibleActions.clear();
    cout << "State" << endl;
    cout << "\t" << resources.minerals << " (+" << (int)(miningSpeed().first*60) << "/min), " << resources.vespene << " (+" << (int)(miningSpeed().second*60) << "/min)," << " food: " << foodAvailable() << " " << endl;
    for (auto& unit : units) {
        cout << "\t" << UnitTypeToName(unit.type) << " x" << unit.units << " (" << unit.busyUnits << " busy)" << endl;
    }
    float nextSignificantEvent = 10000000;
    for (auto& ev : events) {
        if (ev.impactsEconomy())
            nextSignificantEvent = min(nextSignificantEvent, ev.time);
    }
    logBuildOrder(extractBuildOrder(traceStates(shared_from_this())));

    auto currentMiningSpeed = miningSpeed();
    auto parentMiningSpeed = parent != nullptr ? parent->miningSpeed() : make_pair(0.0f,0.0f);

    int totalSupply = foodAvailable();
    int parentTotalSupply = parent != nullptr ? foodAvailable() : 0;


    for (auto& unit : units) {
        if (unit.availableUnits() == 0) continue;

        // TODO: Trim abilities (e.g. remove move abilities which will not be used anyway)
        for (auto ability : unitAbilities(unit.type)) {
            if (contains(blacklistedAbilities, ability)) continue;
            
            auto createdUnit = abilityToUnit(ability);
            if (createdUnit != UNIT_TYPEID::INVALID) {
                auto& abilityData = getAbilityData(ability);
                auto& unitData = getUnitData(createdUnit);

                if (totalSupply < unitData.food_required) {
                    continue;
                }

                if (unitData.tech_requirement != UNIT_TYPEID::INVALID) {
                    if (!hasEquivalentTech(unitData.tech_requirement)) {
                        continue;
                    }
                }

                // if (createdUnit != UNIT_TYPEID::TERRAN_SCV) continue;
                // if (createdUnit != UNIT_TYPEID::TERRAN_SCV && createdUnit != UNIT_TYPEID::TERRAN_ENGINEERINGBAY) continue;

                // TODO: Handle cumulative resource costs for some buildings

                cout << "Potential action " << AbilityTypeToName(ability) << " (costs " << unitData.mineral_cost << "+" << unitData.vespene_cost << ")" << endl;

                // TODO: Needs to be supply edge triggered as well
                // Edge = can do in this state, but couldn't in the parent state (excluding minerals/gas)
                if (unitData.mineral_cost >= resources.minerals || unitData.vespene_cost >= resources.vespene) {
                    // Note: may be infinity if the mining rate is zero
                    float eventTime = time + timeToGetResources(currentMiningSpeed, unitData.mineral_cost, unitData.vespene_cost);
                    cout << "Time: " << eventTime << endl;
                    if (eventTime < nextSignificantEvent) {
                        possibleActions.push_back(make_pair(unit.type, ability));

                        // Do edge triggering, make sure it was *NOT* possible to do the action at the last state, or that the action was done (and we want to repeat it)
                        if (parent != nullptr && !(parentAction.type == BuildActionType::CastAbility && parentAction.ability == ability)) {
                            bool possibleInParent = false;
                            for (auto p : parent->possibleActions) {
                                if (p.first == unit.type && p.second == ability) {
                                    possibleInParent = true;
                                    break;
                                }
                            }
                            if (possibleInParent) {
                                continue;
                            }
                        }

                        // Simulate up to this point
                        shared_ptr<BuildState> newState = make_shared<BuildState>(*this);
                        newState->simulate(eventTime);
                        cout << "After sim time " << newState->time << " " << newState->resources.minerals << "+" << newState->resources.vespene << endl;
                        newState->resources.minerals -= unitData.mineral_cost;
                        newState->resources.vespene -= unitData.vespene_cost;
                        newState->parent = shared_from_this();
                        newState->parentAction = BuildAction();
                        newState->parentAction.type = BuildActionType::CastAbility;
                        newState->parentAction.ability = ability;
                        newState->makeUnitsBusy(unit.type, 1);
                        newState->addEvent(BuildEvent(BuildEventType::FinishedUnit, newState->time + ticksToSeconds(unitData.build_time), unit.type, ability));
                        // cout << "Adding new state at " << newState->time << endl;
                        search.addState(newState);
                    }
                }
            }
        }
    }

    if (events.size() > 0) {
        // Simulate up to this point
        shared_ptr<BuildState> newState = make_shared<BuildState>(*this);
        newState->parent = shared_from_this();
        newState->parentAction = BuildAction();
        newState->parentAction.type = BuildActionType::Wait;
        newState->parentAction.ability = ABILITY_ID::INVALID;
        newState->simulate(events[0].time);
        // cout << "Adding new (noop) state at " << newState->time << endl;
        search.addState(newState);
    }
}

bool BuildEvent::impactsEconomy() const {
    // TODO: Optimize?
    UNIT_TYPEID unit = abilityToUnit(ability);
    return isBasicHarvester(unit) || isStructure(unit) || getUnitData(unit).food_provided > 0;
}

void BuildEvent::apply(BuildState& state) {
    switch (type) {
        case FinishedUnit: {
            UNIT_TYPEID unit = abilityToUnit(ability);
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

// The lower time this state can possibly lead to reaching the target
// TODO: Add in 'steps' to the target?
float lowerBound(const BuildState& state, const BuildTarget& target) {
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

    // Critical path analysis
    float maxBuildTime = 0;
    for (auto& u : remainingUnits) {
        if (u.second > 0) {
            // cout << "Requires unit " << UnitTypeToName(u.first) << endl;
            UNIT_TYPEID type = u.first;
            float criticalPathTime = 0;
            do {
                auto& unitData = getUnitData(type);
                float buildTime = ticksToSeconds(unitData.build_time);
                bool doBreak = false;

                if (type != u.first) {
                    for (auto& u : state.units) {
                        if (u.type == type) {
                            buildTime = 0;
                            doBreak = true;
                            break;
                        }
                        for (auto t : getUnitData(u.type).tech_alias) {
                            if (t == type) {
                                buildTime = 0;
                                doBreak = true;
                                break;
                            }
                        }
                    }

                    if (buildTime > 0) {
                        for (auto& ev : state.events) {
                            if (ev.type == FinishedUnit) {
                                UNIT_TYPEID unit = abilityToUnit(ev.ability);
                                if (unit == type) {
                                    buildTime = 0;
                                    doBreak = true;
                                    break;
                                }
                                for (auto t : getUnitData(unit).tech_alias) {
                                    if (t == type) {
                                        buildTime = min(buildTime, ev.time - state.time);
                                        doBreak = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // cout << "Depends on " << UnitTypeToName(type) << " with " << buildTime << endl;

                criticalPathTime += buildTime;

                // Has to break here in this case to avoid an infinite loop of SCV depends on command center depends on SCV depends on command center ...
                if (isBasicHarvester(type)) {
                    doBreak = true;
                }

                if (doBreak) break;
                type = unitData.tech_requirement;
                if (type == UNIT_TYPEID::INVALID) {
                    // TODO: Handle multiple units case
                    type = abilityToCasterUnit(unitData.ability_id)[0];
                }
            } while (type != UNIT_TYPEID::INVALID);

            maxBuildTime = max(maxBuildTime, criticalPathTime);
        }
    }

    // cout << "Critical path " << maxBuildTime << endl;

    int previousActions = 0;
    shared_ptr<const BuildState> st = state.shared_from_this();
    while(st->parent != nullptr) {
        if (st->parentAction.type == BuildActionType::CastAbility) previousActions++;
        st = st->parent;
    }

    int remainingActions = 0;
    for (auto& u : remainingUnits) {
        remainingActions += u.second;
    }

    /*float requiredMinerals = 0;
    float requiredVespene = 0;
    float maxBuildTime = 0;
    for (auto& u : remainingUnits) {
        if (u.second > 0) {
            auto& unitData = getUnitData(u.first);
            maxBuildTime = max(maxBuildTime, ticksToSeconds(unitData.build_time));
        }
    }*/

    float estimatedTotalTime = max(state.time + maxBuildTime, maxRelevantEventTime);
    float estimatedTotalActions = previousActions + remainingActions;
    return estimatedTotalTime;
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

    int requiredFood = 0;
    for (auto u : target.units) {
        requiredFood += u.second * getUnitData(u.first).food_required;
    }

    // Calculate remaining units that we need to build
    vector<pair<UNIT_TYPEID, int>> remainingUnits = target.units;
    for (auto& u : state.units) {
        requiredFood -= u.units * getUnitData(u.type).food_provided;
        for (auto& u2 : remainingUnits) {
            if (u2.first == u.type) {
                u2 = make_pair(u2.first, max(0, u2.second - u.units));
            }
        }
    }

    for (auto& ev : state.events) {
        if (ev.type == FinishedUnit) {
            UNIT_TYPEID unit = abilityToUnit(ev.ability);
            requiredFood -= 1 * getUnitData(unit).food_provided;

            for (auto& u2 : remainingUnits) {
                if (u2.first == unit && u2.second > 0) {
                    u2 = make_pair(u2.first, u2.second - 1);
                }
            }
        }
    }

    auto supplyU = supplyUnit(target.race);
    auto supplyUnitFood = getUnitData(supplyU).food_provided;
    if (requiredFood > 0)
        remainingUnits.push_back(make_pair(supplyU, (requiredFood + supplyUnitFood - 1) / supplyUnitFood));
    
    return 0;
}

shared_ptr<BuildState> BuildOptimizerSearch::search() {
    float lowestUpperBound = 1000000;
    shared_ptr<BuildState> bestResult = nullptr;
    int numStates = 0;

    while (!que.empty()) {
        numStates++;
        auto p = que.top();
        auto lb = p.first;
        auto state = p.second;
        cout << "Searching state " << numStates << " at " << state->time << " with lb = " << lb << endl;
        que.pop();

        if (lb > lowestUpperBound) {
            // This state cannot possibly be optimal, so skip it
            continue;
        }

        /*float ub = upperBound(state);
        if (ub < lowestUpperBound) {
            lowestUpperBound = ub;
            bestResult = state;
        }*/

        // If it's not done, add more actions
        vector<pair<UNIT_TYPEID, int>> remainingUnits = target.units;
        for (auto& u : state->units) {
            for (auto& u2 : remainingUnits) {
                if (u2.first == u.type) {
                    u2 = make_pair(u2.first, max(0, u2.second - u.units));
                }
            }
        }

        int remaining = 0;
        for (auto& r : remainingUnits) remaining += r.second;

        if (remaining > 0) {
            state->doActions(*this);
        } else {
            cout << "Found optimal state " << " " << state->time << " "  << (state->time == lb) << endl;
            return state;
        }
    }

    return nullptr;
}

void BuildOptimizer::init() {
}

vector<UNIT_TYPEID> BuildOptimizer::calculate_build_order(Race race, const vector<pair<UNIT_TYPEID, int>>& start, const vector<pair<UNIT_TYPEID, int>>& target) {
    // TODO: Constructor
    BuildTarget buildTarget;
    buildTarget.race = race;
    buildTarget.units = target;
    BuildOptimizerSearch search(buildTarget);

    shared_ptr<BuildState> state = make_shared<BuildState>();
    for (auto p : start) {
        state->units.push_back(BuildUnitInfo(p.first, p.second));
    }
    search.addState(state);
    auto optimal = search.search();
    if (optimal == nullptr) {
        return {};
    } else {
        logDetailedBuildOrder(traceStates(optimal));
        return extractBuildOrder(traceStates(optimal));
    }
}

void unitTestBuildOptimizer(BuildOptimizer& optimizer) {
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 1 } });
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 2 } });
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 5 } });
    logBuildOrder(optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_MARINE, 5 }, { UNIT_TYPEID::TERRAN_MEDIVAC, 1 } }));
    // logBuildOrder(optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_MARINE, 5 } }));
}

#endif