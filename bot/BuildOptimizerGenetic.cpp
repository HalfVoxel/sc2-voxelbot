#include "BuildOptimizerGenetic.h"
#include <algorithm>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <stack>
#include <vector>
#include "Bot.h"
#include "utilities/mappings.h"
#include "utilities/predicates.h"
#include "utilities/profiler.h"
#include "utilities/stdutils.h"

using namespace std;
using namespace sc2;

using GeneUnitType = int;

UNIT_TYPEID getSpecificAddonType(UNIT_TYPEID caster, UNIT_TYPEID addon) {
    assert(addon == UNIT_TYPEID::TERRAN_REACTOR || addon == UNIT_TYPEID::TERRAN_TECHLAB);
    switch (caster) {
        case UNIT_TYPEID::TERRAN_BARRACKS:
            return addon == UNIT_TYPEID::TERRAN_TECHLAB ? UNIT_TYPEID::TERRAN_BARRACKSTECHLAB : UNIT_TYPEID::TERRAN_BARRACKSREACTOR;
        case UNIT_TYPEID::TERRAN_FACTORY:
            return addon == UNIT_TYPEID::TERRAN_TECHLAB ? UNIT_TYPEID::TERRAN_FACTORYTECHLAB : UNIT_TYPEID::TERRAN_FACTORYREACTOR;
        case UNIT_TYPEID::TERRAN_STARPORT:
            return addon == UNIT_TYPEID::TERRAN_TECHLAB ? UNIT_TYPEID::TERRAN_STARPORTTECHLAB : UNIT_TYPEID::TERRAN_STARPORTREACTOR;
        default:
            assert(false);
            return UNIT_TYPEID::INVALID;
    }
}

void printBuildOrder(vector<UNIT_TYPEID> buildOrder);

/** Times in the SC2 API are often defined in ticks, instead of seconds.
 * This method assumes the 'Faster' game speed.
 */
float ticksToSeconds(float ticks) {
    return ticks / 22.4f;
}

bool isStructure(UNIT_TYPEID type) {
    // TODO: Cache in mappings?
    return isStructure(getUnitData(type));
}

struct BuildUnitInfo {
    UNIT_TYPEID type;
    UNIT_TYPEID addon;
    int units;
    // E.g. constructing a building, training a unit, etc.
    int busyUnits;

    BuildUnitInfo()
        : type(UNIT_TYPEID::INVALID), addon(UNIT_TYPEID::INVALID), units(0), busyUnits(0) {}
    BuildUnitInfo(UNIT_TYPEID type, UNIT_TYPEID addon, int units)
        : type(type), addon(addon), units(units), busyUnits(0) {}

    int availableUnits() const {
        if (addon == UNIT_TYPEID::TERRAN_REACTOR) {
            return units - busyUnits / 2;
        } else {
            return units - busyUnits;
        }
    }
};

struct BuildResources {
    float minerals;
    float vespene;

    BuildResources(float minerals, float vespene)
        : minerals(minerals), vespene(vespene) {}

    void simulateMining(pair<float, float> miningSpeed, float dt) {
        minerals += miningSpeed.first * dt;
        vespene += miningSpeed.second * dt;
    }
};

struct BuildState;

enum BuildEventType {
    FinishedUnit,
    SpawnLarva,
    MuleTimeout,
};

struct BuildEvent {
    BuildEventType type;
    ABILITY_ID ability;
    UNIT_TYPEID caster;
    UNIT_TYPEID casterAddon;
    float time;

    BuildEvent(BuildEventType type, float time, UNIT_TYPEID caster, ABILITY_ID ability)
        : type(type), ability(ability), caster(caster), casterAddon(UNIT_TYPEID::INVALID), time(time) {}

    bool impactsEconomy() const;
    void apply(BuildState& state);

    bool operator<(const BuildEvent& other) const {
        return time < other.time;
    }
};

struct BuildState {
    float time;

    vector<BuildUnitInfo> units;
    vector<BuildEvent> events;
    BuildResources resources;

    BuildState()
        : time(0), units(), events(), resources(0, 0) {}
    BuildState(vector<pair<UNIT_TYPEID, int>> unitCounts)
        : time(0), units(), events(), resources(0, 0) {
        for (auto u : unitCounts)
            addUnits(u.first, u.second);
    }

    void makeUnitsBusy(UNIT_TYPEID type, UNIT_TYPEID addon, int delta) {
        for (auto& u : units) {
            if (u.type == type && u.addon == addon) {
                u.busyUnits += delta;
                assert(u.busyUnits <= u.units);
                return;
            }
        }
        assert(false);
    }

    void addUnits(UNIT_TYPEID type, int delta) {
        addUnits(type, UNIT_TYPEID::INVALID, delta);
    }

    void addUnits(UNIT_TYPEID type, UNIT_TYPEID addon, int delta) {
        for (auto& u : units) {
            if (u.type == type && u.type == addon) {
                u.units += delta;
                assert(u.availableUnits() >= 0);
                return;
            }
        }

        if (delta > 0)
            units.emplace_back(type, addon, delta);
        else
            assert(false);
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
        mineralMining = min(mineralMining, bases * 24);

        // First 2 harvesters per mineral field yield more minerals than the 3rd one.
        int highYieldHarvesters = min(bases * 8 * 2, mineralMining);
        int lowYieldHarvesters = mineralMining - highYieldHarvesters;

        // TODO: Check units here!
        const float FasterSpeedMultiplier = 1.4f;
        const float LowYieldMineralsPerMinute = 22 / FasterSpeedMultiplier;
        const float HighYieldMineralsPerMinute = 40 / FasterSpeedMultiplier;
        const float VespenePerMinute = 38 / FasterSpeedMultiplier;
        const float MinutesPerSecond = 1 / 60.0f;
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
    }

    // All actions up to and including the end time will have been completed
    void simulate(float endTime) {
        if (endTime <= time)
            return;

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

            if (ev.impactsEconomy()) {
                currentMiningSpeed = miningSpeed();
            } else {
                assert(currentMiningSpeed == miningSpeed());
            }
        }

        events.erase(events.begin(), events.begin() + eventIndex);

        {
            float dt = endTime - time;
            resources.simulateMining(currentMiningSpeed, dt);
            time = endTime;
        }
    }

    bool simulateBuildOrder(vector<UNIT_TYPEID> buildOrder) {
        float lastEventInBuildOrder = 0;
        int buildIndex = -1;
        for (auto unitType : buildOrder) {
            buildIndex++;
            // cout << "Build step " << UnitTypeToName(unitType) << " at " << time << " " << resources.minerals << "+" << resources.vespene << endl;
            while (true) {
                float nextSignificantEvent = numeric_limits<float>::infinity();
                for (auto& ev : events) {
                    if (ev.impactsEconomy()) {
                        nextSignificantEvent = ev.time;
                        break;
                    }
                }

                auto& unitData = getUnitData(unitType);

                if ((unitData.tech_requirement != UNIT_TYPEID::INVALID && !unitData.require_attached && !hasEquivalentTech(unitData.tech_requirement)) || (unitData.food_required > 0 && foodAvailable() < unitData.food_required)) {
                    if (events.empty()) {
                        cout << "No tech"
                             << " at index " << buildIndex << endl;
                        cout << "Requires " << UnitTypeToName(unitData.tech_requirement) << endl;
                        cout << foodAvailable() << " " << unitData.food_required << endl;
                        cout << UnitTypeToName(unitType) << endl;
                        printBuildOrder(buildOrder);
                        exit(1);
                        return false;
                    }
                    simulate(events[0].time);
                    // cout << "No tech, simulating to " << time << endl;
                    continue;
                }

                bool isUnitAddon = isAddon(unitType);

                // TODO: Maybe just use lookup table
                int mineralCost = unitData.mineral_cost;
                int vespeneCost = unitData.vespene_cost;
                UNIT_TYPEID previous = upgradedFrom(unitType);
                if (previous != UNIT_TYPEID::INVALID && !isUnitAddon) {
                    auto& previousUnitData = getUnitData(previous);
                    mineralCost -= previousUnitData.mineral_cost;
                    vespeneCost -= previousUnitData.vespene_cost;
                }

                auto currentMiningSpeed = miningSpeed();
                float eventTime = time + timeToGetResources(currentMiningSpeed, mineralCost, vespeneCost);
                if (eventTime > nextSignificantEvent) {
                    simulate(nextSignificantEvent);
                    // cout << "Too late " << eventTime << " " << nextSignificantEvent << endl;
                    continue;
                }

                // TODO: Need to handle multiple casters case (e.g. need to be able to make SCVs from planetary fortress)
                ABILITY_ID ability = unitData.ability_id;
                assert(abilityToCasterUnit(ability).size() > 0);

                BuildUnitInfo* casterUnit = nullptr;
                UNIT_TYPEID casterUnitType = UNIT_TYPEID::INVALID;
                for (UNIT_TYPEID caster : abilityToCasterUnit(ability)) {
                    // cout << "Caster: " << UnitTypeToName(caster) << " " << UnitTypeToName(unitData.tech_requirement) << endl;
                    for (auto& casterCandidate : units) {
                        // cout << UnitTypeToName(casterCandidate.type) << " " << UnitTypeToName(casterCandidate.addon) << " " << casterCandidate.availableUnits() << " " << unitData.require_attached << endl;
                        if (casterCandidate.type == caster && casterCandidate.availableUnits() > 0 && (!unitData.require_attached || casterCandidate.addon == unitData.tech_requirement)) {
                            // Addons can only be added to units that do not yet have any other addons
                            if (isUnitAddon && casterCandidate.addon != UNIT_TYPEID::INVALID)
                                continue;

                            // Prefer to use casters that do not have addons
                            if (casterUnit == nullptr || casterUnit->addon != UNIT_TYPEID::INVALID) {
                                casterUnit = &casterCandidate;
                                casterUnitType = caster;
                            }
                        }
                    }
                }

                if (casterUnit == nullptr) {
                    if (casterUnitType == UNIT_TYPEID::ZERG_LARVA) {
                        addUnits(UNIT_TYPEID::ZERG_LARVA, 1);
                        continue;
                    }

                    if (events.empty()) {
                        cout << "No possible caster " << UnitTypeToName(unitType) << endl;
                        return false;
                    }
                    simulate(events[0].time);
                    // cout << "No caster, simulating to " << time << endl;
                    continue;
                }

                if (isinf(eventTime)) {
                    cout << "Inf time" << endl;
                    return false;
                }

                simulate(eventTime);

                // simulation may invalidate pointers
                for (auto& casterCandidate : units) {
                    if (casterCandidate.type == casterUnitType && casterCandidate.availableUnits() > 0) {
                        casterUnit = &casterCandidate;
                    }
                }
                assert(casterUnit != nullptr);

                resources.minerals -= mineralCost;
                resources.vespene -= vespeneCost;
                casterUnit->busyUnits++;
                assert(casterUnit->availableUnits() >= 0);
                auto newEvent = BuildEvent(BuildEventType::FinishedUnit, time + ticksToSeconds(unitData.build_time), casterUnit->type, ability);
                newEvent.casterAddon = casterUnit->addon;
                lastEventInBuildOrder = max(lastEventInBuildOrder, newEvent.time);
                addEvent(newEvent);
                break;
            }
        }

        simulate(lastEventInBuildOrder);
        return true;
    }

    // Note that food is a floating point number, zerglings in particular use 0.5 food.
    // It is still safe to work with floating point numbers because they can exactly represent whole numbers and whole numbers + 0.5 exactly up to very large values.
    float foodAvailable() const {
        float totalSupply = 0;
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
            for (auto t : unitData.tech_alias)
                if (t == type)
                    return true;
        }
        return false;
    }
};

bool BuildEvent::impactsEconomy() const {
    // TODO: Optimize?
    UNIT_TYPEID unit = abilityToUnit(ability);
    return isBasicHarvester(unit) || isStructure(unit) || getUnitData(unit).food_provided > 0;
}

void BuildEvent::apply(BuildState& state) {
    switch (type) {
        case FinishedUnit: {
            UNIT_TYPEID unit = abilityToUnit(ability);
            state.makeUnitsBusy(caster, casterAddon, -1);
            if (isAddon(unit)) {
                // Normalize from e.g. TERRAN_BARRACKSTECHLAB to TERRAN_TECHLAB
                unit = simplifyUnitType(unit);
                state.addUnits(caster, unit, 1);
            } else {
                state.addUnits(unit, 1);
            }

            auto upgradedFromUnit = upgradedFrom(unit);
            if (upgradedFromUnit != UNIT_TYPEID::INVALID) {
                state.addUnits(upgradedFromUnit, -1);
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

struct Gene {
    // Indices are into the availableUnitTypes list
    vector<GeneUnitType> buildOrder;

    void mutateMove(float amount, default_random_engine& seed) {
        bernoulli_distribution shouldMutate(amount);
        for (int i = 0; i < buildOrder.size(); i++) {
            if (shouldMutate(seed)) {
                normal_distribution<float> dist(i, buildOrder.size() * 0.25f);
                while (true) {
                    int moveIndex = (int)round(dist(seed));

                    // Try again until we get a number in the right range
                    if (moveIndex < 0 || moveIndex >= buildOrder.size())
                        continue;

                    int s = 0;
                    for (int c : buildOrder)
                        s += c;

                    auto elem = buildOrder[i];
                    // Move element i to the new position by pushing the elements in between one step
                    if (moveIndex < i) {
                        for (int j = i; j > moveIndex; j--)
                            buildOrder[j] = buildOrder[j - 1];
                    } else {
                        for (int j = i; j < moveIndex; j++)
                            buildOrder[j] = buildOrder[j + 1];
                    }
                    buildOrder[moveIndex] = elem;

                    int s2 = 0;
                    for (int c : buildOrder)
                        s2 += c;
                    assert(s == s2);
                    break;
                }
            }
        }
    }

    void mutateAddRemove(float amount, default_random_engine& seed, const vector<int>& actionRequirements, const vector<int>& addableUnits) {
        vector<int> remainingRequirements = actionRequirements;
        for (int i = 0; i < buildOrder.size(); i++) {
            remainingRequirements[buildOrder[i]]--;
        }

        // Remove elements randomly unless that violates the requirements
        bernoulli_distribution shouldRemove(amount);
        for (int i = buildOrder.size() - 1; i >= 0; i--) {
            if (remainingRequirements[buildOrder[i]] < 0 && shouldRemove(seed)) {
                // Remove it!
                remainingRequirements[buildOrder[i]]++;
                buildOrder.erase(buildOrder.begin() + i);
            }
        }

        // Add elements randomly
        bernoulli_distribution shouldAdd(amount);
        for (int i = 0; i < buildOrder.size(); i++) {
            if (shouldAdd(seed)) {
                // Add something!
                uniform_int_distribution<int> dist(0, addableUnits.size() - 1);
                buildOrder.insert(buildOrder.begin() + i, addableUnits[dist(seed)]);
            }
        }

        vector<int> remainingRequirements2 = actionRequirements;
        for (auto type : buildOrder)
            remainingRequirements2[type]--;
        for (auto r : remainingRequirements2)
            assert(r <= 0);
    }

    static Gene crossover(const Gene& parent1, const Gene& parent2, default_random_engine& seed, const vector<int>& actionRequirements) {
        uniform_real_distribution<float> dist(0, 1);
        float split = dist(seed);
        int index1 = min((int)floor(split * parent1.buildOrder.size()), (int)parent1.buildOrder.size());
        int index2 = min((int)floor(split * parent2.buildOrder.size()), (int)parent2.buildOrder.size());

        // Add the elements from parent1
        Gene gene;
        for (int i = 0; i < index1; i++) {
            gene.buildOrder.push_back(parent1.buildOrder[i]);
        }

        // In the middle: insert required actions that are missing from the crossover
        // Calculate any missing requirements
        vector<int> remainingRequirements = actionRequirements;
        for (auto type : gene.buildOrder)
            remainingRequirements[type]--;
        for (int i = index2; i < parent2.buildOrder.size(); i++)
            remainingRequirements[parent2.buildOrder[i]]--;

        for (GeneUnitType i = 0; i < remainingRequirements.size(); i++) {
            for (int j = remainingRequirements[i] - 1; j >= 0; j--)
                gene.buildOrder.push_back(i);
        }

        // Shuffle the elements in the middle that we just added (not the part from parent1)
        shuffle(gene.buildOrder.begin() + index1, gene.buildOrder.end(), seed);

        // Add the elements from parent2
        for (int i = index2; i < parent2.buildOrder.size(); i++) {
            gene.buildOrder.push_back(parent2.buildOrder[i]);
        }
        return gene;
    }

    Gene()
        : buildOrder() {}

    Gene(default_random_engine& seed, vector<int>& actionRequirements) {
        for (GeneUnitType i = 0; i < actionRequirements.size(); i++) {
            for (int j = actionRequirements[i] - 1; j >= 0; j--)
                buildOrder.push_back(i);
        }
        shuffle(buildOrder.begin(), buildOrder.end(), seed);
    }

    static void traceDependencies(const vector<int>& unitCounts, const vector<UNIT_TYPEID>& availableUnitTypes, stack<UNIT_TYPEID>& requirements, UNIT_TYPEID requiredType) {
        // Need to break here to avoid an infinite loop of SCV requires command center requires SCV ...
        if (isBasicHarvester(requiredType))
            return;

        auto& unitData = getUnitData(requiredType);
        if (unitData.tech_requirement != UNIT_TYPEID::INVALID) {
            requiredType = unitData.tech_requirement;
            if (unitData.require_attached) {
                // techlab -> barracks-techlab for example.
                // We need to do this as otherwise the build order becomes ambiguous.
                requiredType = getSpecificAddonType(abilityToCasterUnit(unitData.ability_id)[0], unitData.tech_requirement);

                if (unitCounts[indexOf(availableUnitTypes, requiredType)] == 0) {
                    // Need to add this type to the build order
                    requirements.push(requiredType);
                    // Note: don't trace dependencies for addons as they will only depend on the caster of this unit, which we will already trace.
                    // This is really a bit of a hack to avoid having to prune the requirements for duplicates, but it's good for performance too.
                }
            } else if (unitCounts[indexOf(availableUnitTypes, requiredType)] == 0) {
                // Need to add this type to the build order
                requirements.push(requiredType);
                traceDependencies(unitCounts, availableUnitTypes, requirements, requiredType);
            }
        }

        if (abilityToCasterUnit(unitData.ability_id).size() > 0) {
            bool found = false;
            for (auto possibleCaster : abilityToCasterUnit(unitData.ability_id)) {
                if (unitCounts[indexOf(availableUnitTypes, possibleCaster)] > 0) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                requiredType = abilityToCasterUnit(unitData.ability_id)[0];

                // Ignore larva
                if (requiredType != UNIT_TYPEID::ZERG_LARVA) {
                    requirements.push(requiredType);
                    traceDependencies(unitCounts, availableUnitTypes, requirements, requiredType);
                }
            }
        }
    }

    vector<UNIT_TYPEID> constructBuildOrder(float startingFood, const vector<UNIT_TYPEID>& uniqueStartingUnits, const vector<UNIT_TYPEID>& availableUnitTypes) const {
        vector<int> unitCounts(availableUnitTypes.size());
        // TODO: Should not be unique
        for (auto t : uniqueStartingUnits) {
            unitCounts[indexOf(availableUnitTypes, t)] += 1;
        }
        vector<UNIT_TYPEID> finalBuildOrder;
        float totalFood = startingFood;
        Race race = Race::Terran;
        UNIT_TYPEID currentSupplyUnit = getSupplyUnitForRace(race);
        UNIT_TYPEID currentVespeneHarvester = getVespeneHarvesterForRace(race);
        UNIT_TYPEID currentTownHall = getTownHallForRace(race);

        // Note: stack always starts empty at each iteration, so it could be moved to inside the loop
        // but having it outside avoids some allocations+deallocations.
        stack<UNIT_TYPEID> reqs;

        for (auto type : buildOrder) {
            auto unitType = availableUnitTypes[type];
            reqs.push(unitType);

            // Analyze the prerequisites for the action and add in implicit dependencies
            // (e.g to train a marine, we first need a baracks)
            // TODO: Need more sophisticated tracking because some dependencies can become invalid by other actions
            // (e.g. when building a planetary fortress, a command center is 'used up')
            // auto requiredType = unitType;
            traceDependencies(unitCounts, availableUnitTypes, reqs, unitType);

            while (!reqs.empty()) {
                auto& d = getUnitData(reqs.top());
                // If we don't have enough food, push a supply unit (e.g. supply depot) to the stack
                float foodDelta = d.food_provided - d.food_required;

                // Check which unit (if any) this unit was created from (e.g. command center -> orbital command)
                auto& previous = hasBeen(reqs.top());
                if (previous.size() > 1) {
                    auto& d2 = getUnitData(previous[1]);
                    foodDelta -= (d2.food_provided - d2.food_required);
                }

                // BUILD ADDITIONAL PYLONS
                if (totalFood + foodDelta < 0 && foodDelta < 0) {
                    reqs.push(currentSupplyUnit);
                    continue;
                }

                // Make sure we have a refinery if we need vespene for this unit
                if (d.vespene_cost > 0 && unitCounts[indexOf(availableUnitTypes, currentVespeneHarvester)] == 0) {
                    reqs.push(currentVespeneHarvester);
                    continue;
                }

                // Only allow 2 vespene harvesting buildings per base
                // TODO: Might be better to account for this by having a much lower harvesting rate?
                if (reqs.top() == currentVespeneHarvester) {
                    int numBases = 0;
                    for (int i = 0; i < availableUnitTypes.size(); i++) {
                        if (isTownHall(availableUnitTypes[i]))
                            numBases += unitCounts[i];
                    }

                    if (unitCounts[indexOf(availableUnitTypes, currentVespeneHarvester)] >= numBases * 2) {
                        reqs.push(currentTownHall);
                        continue;
                    }
                }

                if (previous.size() > 1) {
                    int idx = indexOf(availableUnitTypes, previous[1]);
                    assert(unitCounts[idx] > 0);
                    unitCounts[idx]--;
                }

                totalFood += foodDelta;
                finalBuildOrder.push_back(reqs.top());
                unitCounts[indexOf(availableUnitTypes, reqs.top())] += 1;
                reqs.pop();
            }
        }

        return finalBuildOrder;
    }
};

void printBuildOrder(vector<UNIT_TYPEID> buildOrder) {
    cout << "Build order size " << buildOrder.size() << endl;
    for (int i = 0; i < buildOrder.size(); i++) {
        cout << "Step " << i << " " << UnitTypeToName(buildOrder[i]) << endl;
    }
}

float calculateFitness(const BuildState& startState, const vector<UNIT_TYPEID> uniqueStartingUnits, const vector<UNIT_TYPEID>& availableUnitTypes, const Gene& gene) {
    BuildState state = startState;
    // auto order = gene.constructBuildOrder(startState.foodAvailable(), uniqueStartingUnits, availableUnitTypes);
    // cout << "Build order size " << order.size() << endl;
    // for (int i = 0; i < order.size(); i++) {
    // cout << "Step " << i << " " << UnitTypeToName(order[i]) << endl;
    // }
    // printBuildOrder(gene.constructBuildOrder(startState.foodAvailable(), uniqueStartingUnits, availableUnitTypes));
    if (!state.simulateBuildOrder(gene.constructBuildOrder(startState.foodAvailable(), uniqueStartingUnits, availableUnitTypes))) {
        // cout << "Failed" << endl;
        return -100000;
    }
    // cout << "Time " << state.time << endl;
    return -state.time + (state.resources.minerals + 2 * state.resources.vespene) * 0.05;
}

Gene locallyOptimizeGene(const BuildState& startState, const vector<UNIT_TYPEID> uniqueStartingUnits, const vector<UNIT_TYPEID>& availableUnitTypes, const Gene& gene) {
    float startFitness = calculateFitness(startState, uniqueStartingUnits, availableUnitTypes, gene);
    float fitness = startFitness;
    Gene newGene = gene;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < newGene.buildOrder.size() - 1; j++) {
            if (newGene.buildOrder[j] != newGene.buildOrder[j + 1]) {
                swap(newGene.buildOrder[j], newGene.buildOrder[j + 1]);
                float newFitness = calculateFitness(startState, uniqueStartingUnits, availableUnitTypes, newGene);

                if (newFitness > fitness) {
                    fitness = newFitness;
                } else {
                    // Revert swap
                    swap(newGene.buildOrder[j], newGene.buildOrder[j + 1]);
                }
            }
        }
    }

    cout << "Optimized from " << startFitness << " to " << fitness << endl;
    return newGene;
}

vector<UNIT_TYPEID> unitTypesZergEconomic = {
    UNIT_TYPEID::ZERG_BANELINGNEST,
    UNIT_TYPEID::ZERG_DRONE,
    UNIT_TYPEID::ZERG_DRONE,
    UNIT_TYPEID::ZERG_DRONE,
    UNIT_TYPEID::ZERG_DRONE,
    UNIT_TYPEID::ZERG_DRONE,
    UNIT_TYPEID::ZERG_EVOLUTIONCHAMBER,
    UNIT_TYPEID::ZERG_EXTRACTOR,
    UNIT_TYPEID::ZERG_GREATERSPIRE,
    UNIT_TYPEID::ZERG_HATCHERY,
    UNIT_TYPEID::ZERG_HIVE,
    UNIT_TYPEID::ZERG_HYDRALISKDEN,
    UNIT_TYPEID::ZERG_INFESTATIONPIT,
    UNIT_TYPEID::ZERG_LAIR,
    UNIT_TYPEID::ZERG_LURKERDENMP,
    UNIT_TYPEID::ZERG_NYDUSCANAL,
    UNIT_TYPEID::ZERG_NYDUSNETWORK,
    UNIT_TYPEID::ZERG_QUEEN,
    UNIT_TYPEID::ZERG_SPAWNINGPOOL,
    UNIT_TYPEID::ZERG_SPIRE,
    UNIT_TYPEID::ZERG_ULTRALISKCAVERN,
};

vector<UNIT_TYPEID> unitTypesZerg = {
    UNIT_TYPEID::ZERG_BANELING,
    UNIT_TYPEID::ZERG_BANELINGNEST,
    UNIT_TYPEID::ZERG_BROODLORD,
    UNIT_TYPEID::ZERG_CORRUPTOR,
    UNIT_TYPEID::ZERG_DRONE,
    UNIT_TYPEID::ZERG_EVOLUTIONCHAMBER,
    UNIT_TYPEID::ZERG_EXTRACTOR,
    UNIT_TYPEID::ZERG_GREATERSPIRE,
    UNIT_TYPEID::ZERG_HATCHERY,
    UNIT_TYPEID::ZERG_HIVE,
    UNIT_TYPEID::ZERG_HYDRALISK,
    UNIT_TYPEID::ZERG_HYDRALISKDEN,
    UNIT_TYPEID::ZERG_INFESTATIONPIT,
    UNIT_TYPEID::ZERG_INFESTOR,
    UNIT_TYPEID::ZERG_LAIR,
    UNIT_TYPEID::ZERG_LURKERDENMP,
    // UNIT_TYPEID::ZERG_LURKERMP,
    UNIT_TYPEID::ZERG_MUTALISK,
    UNIT_TYPEID::ZERG_NYDUSCANAL,
    UNIT_TYPEID::ZERG_NYDUSNETWORK,
    UNIT_TYPEID::ZERG_OVERLORD,
    UNIT_TYPEID::ZERG_OVERLORDTRANSPORT,
    UNIT_TYPEID::ZERG_OVERSEER,
    UNIT_TYPEID::ZERG_QUEEN,
    UNIT_TYPEID::ZERG_RAVAGER,
    UNIT_TYPEID::ZERG_ROACH,
    UNIT_TYPEID::ZERG_ROACHWARREN,
    UNIT_TYPEID::ZERG_SPAWNINGPOOL,
    UNIT_TYPEID::ZERG_SPINECRAWLER,
    UNIT_TYPEID::ZERG_SPIRE,
    UNIT_TYPEID::ZERG_SPORECRAWLER,
    UNIT_TYPEID::ZERG_SWARMHOSTMP,
    UNIT_TYPEID::ZERG_ULTRALISK,
    UNIT_TYPEID::ZERG_ULTRALISKCAVERN,
    UNIT_TYPEID::ZERG_VIPER,
    UNIT_TYPEID::ZERG_ZERGLING,
};

vector<UNIT_TYPEID> unitTypesTerranEconomic = {
    UNIT_TYPEID::TERRAN_ARMORY,
    UNIT_TYPEID::TERRAN_BARRACKS,
    UNIT_TYPEID::TERRAN_BARRACKSREACTOR,
    UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
    UNIT_TYPEID::TERRAN_COMMANDCENTER,
    UNIT_TYPEID::TERRAN_ENGINEERINGBAY,
    UNIT_TYPEID::TERRAN_FACTORY,
    UNIT_TYPEID::TERRAN_FACTORYREACTOR,
    UNIT_TYPEID::TERRAN_FACTORYTECHLAB,
    UNIT_TYPEID::TERRAN_FUSIONCORE,
    UNIT_TYPEID::TERRAN_GHOSTACADEMY,
    // UNIT_TYPEID::TERRAN_MULE,
    UNIT_TYPEID::TERRAN_ORBITALCOMMAND,
    UNIT_TYPEID::TERRAN_PLANETARYFORTRESS,
    UNIT_TYPEID::TERRAN_REFINERY,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_STARPORT,
    UNIT_TYPEID::TERRAN_STARPORTREACTOR,
    UNIT_TYPEID::TERRAN_STARPORTTECHLAB,
    UNIT_TYPEID::TERRAN_SUPPLYDEPOT,
};

vector<UNIT_TYPEID> unitTypesTerran = {
    UNIT_TYPEID::TERRAN_ARMORY,
    UNIT_TYPEID::TERRAN_BANSHEE,
    UNIT_TYPEID::TERRAN_BARRACKS,
    UNIT_TYPEID::TERRAN_BARRACKSREACTOR,
    UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
    UNIT_TYPEID::TERRAN_BATTLECRUISER,
    UNIT_TYPEID::TERRAN_BUNKER,
    UNIT_TYPEID::TERRAN_COMMANDCENTER,
    UNIT_TYPEID::TERRAN_CYCLONE,
    UNIT_TYPEID::TERRAN_ENGINEERINGBAY,
    UNIT_TYPEID::TERRAN_FACTORY,
    UNIT_TYPEID::TERRAN_FACTORYREACTOR,
    UNIT_TYPEID::TERRAN_FACTORYTECHLAB,
    UNIT_TYPEID::TERRAN_FUSIONCORE,
    UNIT_TYPEID::TERRAN_GHOST,
    UNIT_TYPEID::TERRAN_GHOSTACADEMY,
    UNIT_TYPEID::TERRAN_HELLION,
    UNIT_TYPEID::TERRAN_HELLIONTANK,
    UNIT_TYPEID::TERRAN_LIBERATOR,
    UNIT_TYPEID::TERRAN_MARAUDER,
    UNIT_TYPEID::TERRAN_MARINE,
    UNIT_TYPEID::TERRAN_MEDIVAC,
    UNIT_TYPEID::TERRAN_MISSILETURRET,
    UNIT_TYPEID::TERRAN_MULE,
    UNIT_TYPEID::TERRAN_ORBITALCOMMAND,
    UNIT_TYPEID::TERRAN_PLANETARYFORTRESS,
    UNIT_TYPEID::TERRAN_RAVEN,
    UNIT_TYPEID::TERRAN_REAPER,
    UNIT_TYPEID::TERRAN_REFINERY,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SENSORTOWER,
    UNIT_TYPEID::TERRAN_SIEGETANK,
    UNIT_TYPEID::TERRAN_STARPORT,
    UNIT_TYPEID::TERRAN_STARPORTREACTOR,
    UNIT_TYPEID::TERRAN_STARPORTTECHLAB,
    UNIT_TYPEID::TERRAN_SUPPLYDEPOT,
    UNIT_TYPEID::TERRAN_THOR,
    UNIT_TYPEID::TERRAN_VIKINGFIGHTER,
    UNIT_TYPEID::TERRAN_WIDOWMINE,
};

void findBestCompositionGenetic(const vector<pair<UNIT_TYPEID, int>>& startingUnits, const vector<pair<UNIT_TYPEID, int>>& target) {
    BuildState startState;
    vector<UNIT_TYPEID> uniqueStartingUnits;
    for (auto p : startingUnits) {
        startState.addUnits(p.first, p.second);
        uniqueStartingUnits.push_back(p.first);
    }

    const vector<UNIT_TYPEID>& availableUnitTypes = unitTypesTerran;

    vector<int> actionRequirements(availableUnitTypes.size());
    for (int i = 0; i < actionRequirements.size(); i++) {
        UNIT_TYPEID type = availableUnitTypes[i];
        int count = 0;
        for (auto p : target)
            if (p.first == type)
                count += p.second;
        for (auto p : startingUnits)
            if (p.first == type)
                count -= p.second;
        actionRequirements[i] = max(0, count);
    }

    vector<int> economicUnits;
    for (auto u : unitTypesTerranEconomic) {
        for (int i = 0; i < availableUnitTypes.size(); i++)
            if (availableUnitTypes[i] == u)
                economicUnits.push_back(i);
    }

    Stopwatch watch;

    const int POOL_SIZE = 25;
    const float mutationRateAddRemove = 0.05f;
    const float mutationRateMove = 0.05f;
    vector<Gene> generation(POOL_SIZE);
    default_random_engine rnd(time(0));
    for (int i = 0; i < POOL_SIZE; i++) {
        generation[i] = Gene(rnd, actionRequirements);
    }
    for (int i = 0; i < 200; i++) {
        vector<float> fitness(generation.size());
        vector<int> indices(generation.size());
        for (int j = 0; j < generation.size(); j++) {
            indices[j] = j;
            fitness[j] = calculateFitness(startState, uniqueStartingUnits, availableUnitTypes, generation[j]);
        }

        sortByValueDescending<int, float>(indices, [=](int index) { return fitness[index]; });
        vector<Gene> nextGeneration;
        // Add the N best performing genes
        for (int j = 0; j < 5; j++) {
            nextGeneration.push_back(generation[indices[j]]);
        }
        // Add a random one as well
        nextGeneration.push_back(generation[uniform_int_distribution<int>(0, indices.size() - 1)(rnd)]);

        if ((i % 50) == 0 && i != 0) {
            for (auto& g : nextGeneration) {
                g = locallyOptimizeGene(startState, uniqueStartingUnits, availableUnitTypes, g);
            }
        }

        uniform_int_distribution<int> randomParentIndex(0, nextGeneration.size() - 1);
        while (nextGeneration.size() < POOL_SIZE) {
            nextGeneration.push_back(Gene::crossover(generation[randomParentIndex(rnd)], generation[randomParentIndex(rnd)], rnd, actionRequirements));
        }

        // Note: do not mutate the first gene
        bernoulli_distribution moveMutation(0.5);
        for (int i = 1; i < nextGeneration.size(); i++) {
            nextGeneration[i].mutateMove(mutationRateMove, rnd);
            nextGeneration[i].mutateAddRemove(mutationRateAddRemove, rnd, actionRequirements, economicUnits);
        }

        swap(generation, nextGeneration);

        cout << "Best fitness " << fitness[indices[0]] << endl;
        // calculateFitness(startState, uniqueStartingUnits, availableUnitTypes, generation[indices[0]]);
    }

    printBuildOrder(generation[0].constructBuildOrder(startState.foodAvailable(), uniqueStartingUnits, availableUnitTypes));
}

void unitTestBuildOptimizer() {
    // assert(BuildState({{ UNIT_TYPEID::ZERG_HATCHERY, 1 }, { UNIT_TYPEID::ZERG_DRONE, 12 }}).simulateBuildOrder({ UNIT_TYPEID::ZERG_SPAWNINGPOOL, UNIT_TYPEID::ZERG_ZERGLING }));

    // findBestCompositionGenetic({ { UNIT_TYPEID::ZERG_HATCHERY, 1 }, { UNIT_TYPEID::ZERG_DRONE, 12 } }, { { UNIT_TYPEID::ZERG_HATCHERY, 1 }, { UNIT_TYPEID::ZERG_ZERGLING, 12 }, { UNIT_TYPEID::ZERG_MUTALISK, 20 }, { UNIT_TYPEID::ZERG_INFESTOR, 1 } });
    for (int i = 0; i < 1; i++) {
        findBestCompositionGenetic({ { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_VIKINGFIGHTER, 2 }, { UNIT_TYPEID::TERRAN_MEDIVAC, 3 }, { UNIT_TYPEID::TERRAN_BANSHEE, 3 }, { UNIT_TYPEID::TERRAN_MARINE, 20 }, { UNIT_TYPEID::TERRAN_BUNKER, 1 } });
        // findBestCompositionGenetic({ { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_BARRACKSREACTOR, 0 }, { UNIT_TYPEID::TERRAN_MARINE, 30 }, { UNIT_TYPEID::TERRAN_MARAUDER, 0 } });
    }
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 1 } });
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 2 } });
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 5 } });
    // logBuildOrder(optimizer.calculate_build_order();
    // logBuildOrder(optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_MARINE, 5 } }));
}
