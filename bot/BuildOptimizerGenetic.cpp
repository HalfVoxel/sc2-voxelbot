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

void printBuildOrder(vector<UNIT_TYPEID> buildOrder);

void BuildState::makeUnitsBusy(UNIT_TYPEID type, UNIT_TYPEID addon, int delta) {
    if (delta == 0) return;

    for (auto& u : units) {
        if (u.type == type && u.addon == addon) {
            u.busyUnits += delta;
            assert(u.availableUnits() >= 0);
            assert(u.busyUnits >= 0);
            return;
        }
    }
    assert(false);
}

void BuildState::addUnits(UNIT_TYPEID type, int delta) {
    addUnits(type, UNIT_TYPEID::INVALID, delta);
}

void BuildState::addUnits(UNIT_TYPEID type, UNIT_TYPEID addon, int delta) {
    if (delta == 0) return;

    for (auto& u : units) {
        if (u.type == type && u.addon == addon) {
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

pair<float, float> BuildState::miningSpeed() const {
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
    const float LowYieldMineralsPerMinute = 22 * FasterSpeedMultiplier;
    const float HighYieldMineralsPerMinute = 40 * FasterSpeedMultiplier;
    const float VespenePerMinute = 38 * FasterSpeedMultiplier;
    const float MinutesPerSecond = 1 / 60.0f;
    float mineralsPerSecond = (lowYieldHarvesters * LowYieldMineralsPerMinute + highYieldHarvesters * HighYieldMineralsPerMinute) * MinutesPerSecond;
    float vespenePerSecond = vespeneMining * VespenePerMinute * MinutesPerSecond;
    return make_pair(mineralsPerSecond, vespenePerSecond);
}

float BuildState::timeToGetResources(pair<float, float> miningSpeed, float mineralCost, float vespeneCost) const {
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

void BuildState::addEvent(BuildEvent event) {
    // TODO: Insertion sort or something
    events.push_back(event);
    sort(events.begin(), events.end());
}

// All actions up to and including the end time will have been completed
void BuildState::simulate(float endTime) {
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

bool BuildState::simulateBuildOrder(vector<UNIT_TYPEID> buildOrder, function<void(int)> callback) {
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
                    cout << "Current unit counts:" << endl;
                    for (auto u : units) {
                        cout << UnitTypeToName(u.type) << " " << UnitTypeToName(u.addon) << " " << u.units << endl;
                    }
                    __builtin_trap();
                    exit(1);
                    return false;
                }

                // cout << UnitTypeToName(unitType) << " Waiting for tech" << endl;
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
                // cout << UnitTypeToName(unitType) << " Waiting for significant event" << endl;
                simulate(nextSignificantEvent);
                // cout << "Too late " << eventTime << " " << nextSignificantEvent << endl;
                continue;
            }

            // TODO: Need to handle multiple casters case (e.g. need to be able to make SCVs from planetary fortress)
            ABILITY_ID ability = unitData.ability_id;
            assert(abilityToCasterUnit(ability).size() > 0);

            BuildUnitInfo* casterUnit = nullptr;
            UNIT_TYPEID casterUnitType = UNIT_TYPEID::INVALID;
            UNIT_TYPEID casterAddonType = UNIT_TYPEID::INVALID;
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
                            casterAddonType = casterCandidate.addon;
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
                    for (auto& casterCandidate : units) {
                        cout << "Caster: " << UnitTypeToName(casterCandidate.type) << " " << casterCandidate.units << "/" << casterCandidate.availableUnits() << " " << UnitTypeToName(casterCandidate.addon) << endl;
                    }
                    exit(1);
                    return false;
                }

                // cout << UnitTypeToName(unitType) << " Waiting for caster" << endl;
                simulate(events[0].time);
                // cout << "No caster, simulating to " << time << endl;
                continue;
            }

            if (isinf(eventTime)) {
                // This can happen in some cases.
                // Most common is when the unit requires vespene gas, but the player only has 1 scv and that one will be allocated to minerals.

                /*cout << "Inf time" << endl;
                cout << foodAvailable() << " " << unitData.food_required << endl;
                cout << UnitTypeToName(unitType) << endl;
                printBuildOrder(buildOrder);
                cout << "Current unit counts:" << endl;
                for (auto u : units) {
                    cout << UnitTypeToName(u.type) << " " << UnitTypeToName(u.addon) << " " << u.units << endl;
                }
                __builtin_trap();*/
                return false;
            }

            simulate(eventTime);

            // simulation may invalidate pointers
            for (auto& casterCandidate : units) {
                if (casterCandidate.type == casterUnitType && casterCandidate.addon == casterAddonType && casterCandidate.availableUnits() > 0) {
                    casterUnit = &casterCandidate;
                }
            }
            assert(casterUnit != nullptr);

            if (callback != nullptr) callback(buildIndex);
            resources.minerals -= mineralCost;
            resources.vespene -= vespeneCost;
            casterUnit->busyUnits++;
            assert(casterUnit->availableUnits() >= 0);
            float buildTime = ticksToSeconds(unitData.build_time);

            // Compensate for workers having to move to the building location
            if (isStructure(unitType)) {
                buildTime += 4;
            }

            auto newEvent = BuildEvent(BuildEventType::FinishedUnit, time + buildTime, casterUnit->type, ability);
            newEvent.casterAddon = casterUnit->addon;
            lastEventInBuildOrder = max(lastEventInBuildOrder, newEvent.time);
            // cout << "Event time " << newEvent.time << endl;
            // cout << "Caster " << UnitTypeToName(casterUnit->type) << " " << UnitTypeToName(casterUnit->addon) << " " << casterUnit->availableUnits() << " " << casterUnit->units << endl;
            // cout << "Current unit counts:" << endl;
            // for (auto u : units) {
            //     cout << UnitTypeToName(u.type) << " " << UnitTypeToName(u.addon) << " " << u.units << endl;
            // }
            addEvent(newEvent);
            break;
        }
    }

    simulate(lastEventInBuildOrder);
    return true;
}

// Note that food is a floating point number, zerglings in particular use 0.5 food.
// It is still safe to work with floating point numbers because they can exactly represent whole numbers and whole numbers + 0.5 exactly up to very large values.
float BuildState::foodAvailable() const {
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

    // Not necessarily true in all game states
    // assert(totalSupply >= 0);
    return totalSupply;
}

bool BuildState::hasEquivalentTech(UNIT_TYPEID type) const {
    for (auto& unit : units) {
        auto& unitData = getUnitData(unit.type);
        if (unit.units > 0) {
            if (unit.type == type) {
                return true;
            }
            for (auto t : unitData.tech_alias)
                if (t == type)
                    return true;
        }
    }
    return false;
}

bool BuildEvent::impactsEconomy() const {
    // TODO: Optimize?
    // TODO: isStructure is very aggressive, isTownHall is more appropriate?
    UNIT_TYPEID unit = abilityToUnit(ability);
    return isBasicHarvester(unit) || isBasicHarvester(caster) || isStructure(unit) || getUnitData(unit).food_provided > 0;
}

void BuildEvent::apply(BuildState& state) {
    switch (type) {
        case FinishedUnit: {
            UNIT_TYPEID unit = abilityToUnit(ability);
            state.makeUnitsBusy(caster, casterAddon, -1);
            if (isAddon(unit)) {
                // Normalize from e.g. TERRAN_BARRACKSTECHLAB to TERRAN_TECHLAB
                state.addUnits(caster, simplifyUnitType(unit), 1);
            } else {
                state.addUnits(unit, 1);
            }

            auto upgradedFromUnit = upgradedFrom(unit);
            if (upgradedFromUnit != UNIT_TYPEID::INVALID) {
                state.addUnits(upgradedFromUnit, casterAddon, -1);
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

    void validate(const vector<int>& actionRequirements) const {
#if DEBUG
        vector<int> remainingRequirements = actionRequirements;
        for (auto type : buildOrder)
            remainingRequirements[type]--;
        for (auto r : remainingRequirements)
            assert(r <= 0);
#endif
    }

    void mutateMove(float amount, const vector<int>& actionRequirements, default_random_engine& seed) {
        validate(actionRequirements);

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

        validate(actionRequirements);
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

        validate(actionRequirements);
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


        gene.validate(actionRequirements);

        return gene;
    }

    Gene()
        : buildOrder() {}

    Gene(default_random_engine& seed, const vector<int>& actionRequirements) {
        for (GeneUnitType i = 0; i < actionRequirements.size(); i++) {
            for (int j = actionRequirements[i] - 1; j >= 0; j--)
                buildOrder.push_back(i);
        }
        shuffle(buildOrder.begin(), buildOrder.end(), seed);
    }

    Gene(const vector<UNIT_TYPEID>& seedBuildOrder, const vector<UNIT_TYPEID>& availableUnitTypes, const vector<int>& actionRequirements) {
        vector<int> remainingRequirements = actionRequirements;
        for (auto u : seedBuildOrder) {
            GeneUnitType type = indexOf(availableUnitTypes, u);
            buildOrder.push_back(type);
            remainingRequirements[type]--;
        }
        for (GeneUnitType i = 0; i < remainingRequirements.size(); i++) {
            int r = remainingRequirements[i];
            for (auto j = 0; j < r; j++) {
                buildOrder.push_back(i);
            }
        }
    }

    static void traceDependencies(const vector<int>& unitCounts, const vector<UNIT_TYPEID>& availableUnitTypes, stack<UNIT_TYPEID>& requirements, UNIT_TYPEID requiredType) {
        // Need to break here to avoid an infinite loop of SCV requires command center requires SCV ...
        if (isBasicHarvester(requiredType))
            return;

        auto& unitData = getUnitData(requiredType);
        if (unitData.tech_requirement != UNIT_TYPEID::INVALID) {
            requiredType = unitData.tech_requirement;

            // Check if the tech requirement is an addon
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

    vector<UNIT_TYPEID> constructBuildOrder(Race race, float startingFood, const vector<int>& startingUnitCounts, const vector<int>& startingAddonCountPerUnitType, const vector<UNIT_TYPEID>& availableUnitTypes) const {

        vector<int> unitCounts = startingUnitCounts;
        vector<int> addonCountPerUnitType = startingAddonCountPerUnitType;
        assert(unitCounts.size() == availableUnitTypes.size());
        vector<UNIT_TYPEID> finalBuildOrder;
        float totalFood = startingFood;
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
            if (false && (unitType == UNIT_TYPEID::TERRAN_STARPORT || unitType == UNIT_TYPEID::TERRAN_FACTORY || true)) {
                auto& unitData = getUnitData(unitType);
                cout << "A " << unitData.require_attached << " " << UnitTypeToName(unitData.tech_requirement) << endl;
                if (!isAddon(unitData.tech_requirement) && unitData.tech_requirement != UNIT_TYPEID::INVALID) cout << "Count: " << unitCounts[indexOf(availableUnitTypes, (UNIT_TYPEID)unitData.tech_requirement)] << endl;

                cout << "Build order!" << endl;
                for (int i = 0; i < startingUnitCounts.size(); i++) {
                    assert(startingUnitCounts[i] >= 0);
                    if (startingUnitCounts[i] > 0) cout << "Starting unit " << UnitTypeToName(availableUnitTypes[i]) << " " << startingUnitCounts[i] << endl;
                }
                for (auto b : buildOrder) {
                    cout << "B " << UnitTypeToName(availableUnitTypes[b]) << endl;
                }


                for (int i = 0; i < unitCounts.size(); i++) {
                    assert(unitCounts[i] >= 0);
                    cout << "Current unit " << UnitTypeToName(availableUnitTypes[i]) << " " << unitCounts[i] << endl;
                }

                auto r2 = reqs;
                while (!r2.empty()) {
                    cout << "Req: " << UnitTypeToName(r2.top()) << endl;
                    r2.pop();
                }
            }

            while (!reqs.empty()) {
                auto requirement = reqs.top();
                auto& d = getUnitData(requirement);
                // If we don't have enough food, push a supply unit (e.g. supply depot) to the stack
                float foodDelta = d.food_provided - d.food_required;

                // Check which unit (if any) this unit was created from (e.g. command center -> orbital command)
                auto& previous = hasBeen(requirement);
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
                if (requirement == currentVespeneHarvester) {
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

                // Addons should always list the original building in the previous list
                assert(!isAddon(requirement) || previous.size() > 1);

                if (previous.size() > 1) {
                    int idx = indexOf(availableUnitTypes, previous[1]);
                    assert(unitCounts[idx] > 0);
                    if (isAddon(requirement)) {
                        // Try to mark another building has having an addon
                        if (addonCountPerUnitType[idx] < unitCounts[idx]) {
                            addonCountPerUnitType[idx]++;
                        } else {
                            // If there are no possible such buildings, then we need to add a new one of those buildings
                            reqs.push(previous[1]);
                            traceDependencies(unitCounts, availableUnitTypes, reqs, previous[1]);
                            continue;
                        }
                    } else {
                        // Remove the previous unit if this is an upgrade (e.g. command center -> planetary fortress)
                        // However make sure not to do it for addons, as the original building is still kept in that case
                        unitCounts[idx]--;
                    }
                }

                totalFood += foodDelta;
                finalBuildOrder.push_back(requirement);
                unitCounts[indexOf(availableUnitTypes, requirement)] += 1;
                reqs.pop();
            }
        }

        return finalBuildOrder;
    }
};

void printBuildOrderDetailed(const BuildState& startState, vector<UNIT_TYPEID> buildOrder) {
    BuildState state = startState;
    cout << "Starting units" << endl;
    for (auto u : startState.units) {
        cout << "\t" << u.units << "x" << " " << UnitTypeToName(u.type);
        if (u.addon != UNIT_TYPEID::INVALID) cout << " + " << UnitTypeToName(u.addon);
        cout << endl;
    }
    cout << "Build order size " << buildOrder.size() << endl;
    state.simulateBuildOrder(buildOrder, [&](int i) {
        cout << "Step " << i << "\t" << (int)(state.time/60.0f) << ":" << (int)(fmod(state.time, 60.0f)) << "\t" << UnitTypeToName(buildOrder[i]) << " " << "food: " << state.foodAvailable() << " " << (int)state.resources.minerals << "+" << (int)state.resources.vespene << endl;
    });

    cout << "Finished at " << (int)(state.time/60.0f) << ":" << (int)(fmod(state.time, 60.0f)) << endl;
}

void printBuildOrder(vector<UNIT_TYPEID> buildOrder) {
    cout << "Build order size " << buildOrder.size() << endl;
    for (int i = 0; i < buildOrder.size(); i++) {
        cout << "Step " << i << " " << UnitTypeToName(buildOrder[i]) << endl;
    }
}

float calculateFitness(const BuildState& startState, const vector<int>& startingUnitCounts, const vector<int>& startingAddonCountPerUnitType, const vector<UNIT_TYPEID>& availableUnitTypes, const Gene& gene) {
    BuildState state = startState;
    // auto order = gene.constructBuildOrder(startState.foodAvailable(), uniqueStartingUnits, availableUnitTypes);
    // cout << "Build order size " << order.size() << endl;
    // for (int i = 0; i < order.size(); i++) {
    // cout << "Step " << i << " " << UnitTypeToName(order[i]) << endl;
    // }
    // printBuildOrder(gene.constructBuildOrder(startState.foodAvailable(), uniqueStartingUnits, availableUnitTypes));
    vector<float> finishedTimes;
    auto buildOrder = gene.constructBuildOrder(startState.race, startState.foodAvailable(), startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes);
    if (!state.simulateBuildOrder(buildOrder, [&] (int index) {
        finishedTimes.push_back(state.time);
    })) {
        // cout << "Failed" << endl;
        return -100000;
    }

    // Score between 0 and 1 where
    // 0 => all units are built right at the end
    // 1 => all units are built right at the start
    float ginoScore = 0;
    float avgTime = 0;
    float totalWeight = 0;
    for (int i = 0; i < finishedTimes.size(); i++) {
        float t = finishedTimes[i];
        ginoScore += (state.time - t)*(state.time - t)*0.5f;
        float w = isArmy(buildOrder[i]) ? 1.0f : 0.1f;
        totalWeight += w;
        avgTime += w*(t + 20); // +20 to take into account that we want the finished time of the unit, but we only have the start time
    }
    ginoScore /= state.time * state.time * finishedTimes.size();

    // A gino score of 0 multiplies the time by 2 essentially
    float ginoMultiplier = 2.0f / (1.0f + ginoScore);
    if (finishedTimes.size() == 0) {
        avgTime = state.time;
    } else {
        avgTime /= totalWeight;
    }


    // cout << "Time " << state.time << endl;
    // return -state.time;

    // Simulate until at least the 2 minutes mark, this ensures that the agent will do some economic stuff if nothing else
    state.simulate(60*2);

    auto miningSpeed = state.miningSpeed();
    // return -state.time * ginoMultiplier + (state.resources.minerals + 2 * state.resources.vespene) * 0.001 + (miningSpeed.first + 2 * miningSpeed.second) * 60 * 0.005;
    return -max(avgTime*2, 2*60.0f) + (state.resources.minerals + 2 * state.resources.vespene) * 0.001 + (miningSpeed.first + 2 * miningSpeed.second) * 60 * 0.005;
}

Gene locallyOptimizeGene(const BuildState& startState, const vector<int>& startingUnitCounts, const vector<int>& startingAddonCountPerUnitType, const vector<UNIT_TYPEID>& availableUnitTypes, const vector<int>& actionRequirements, const Gene& gene) {
    vector<int> currentActionRequirements = actionRequirements;
    for (auto b : gene.buildOrder) currentActionRequirements[b]--;

    float startFitness = calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, gene);
    float fitness = startFitness;
    Gene newGene = gene;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < newGene.buildOrder.size(); j++) {

            if (j == newGene.buildOrder.size() - 1 || newGene.buildOrder[j] != newGene.buildOrder[j + 1]) {
                // Check if the item is non-essential
                if (currentActionRequirements[newGene.buildOrder[j]] < 0) {
                    // Try removing
                    auto orig = newGene.buildOrder[j];
                    newGene.buildOrder.erase(newGene.buildOrder.begin() + j);
                    float newFitness = calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, newGene);
                    if (newFitness > fitness) {
                        currentActionRequirements[orig] += 1;
                        fitness = newFitness;
                        j--;
                        continue;
                    } else {
                        // Revert erase
                        newGene.buildOrder.insert(newGene.buildOrder.begin() + j, orig);
                    }
                }

                // Try swapping
                {
                    swap(newGene.buildOrder[j], newGene.buildOrder[j + 1]);
                    float newFitness = calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, newGene);

                    if (newFitness > fitness) {
                        fitness = newFitness;
                    } else {
                        // Revert swap
                        swap(newGene.buildOrder[j], newGene.buildOrder[j + 1]);
                    }
                }
            }
        }

        // cout << "[" << i << "] " << "Optimized from " << startFitness << " to " << fitness << endl;
    }

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

static vector<UNIT_TYPEID> unitTypesTerran = {
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

vector<UNIT_TYPEID> unitTypesProtoss = {
    UNIT_TYPEID::PROTOSS_ADEPT,
    // UNIT_TYPEID::PROTOSS_ADEPTPHASESHIFT,
    UNIT_TYPEID::PROTOSS_ARCHON,
    UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_CARRIER,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
    UNIT_TYPEID::PROTOSS_DARKSHRINE,
    UNIT_TYPEID::PROTOSS_DARKTEMPLAR,
    UNIT_TYPEID::PROTOSS_DISRUPTOR,
    // UNIT_TYPEID::PROTOSS_DISRUPTORPHASED,
    UNIT_TYPEID::PROTOSS_FLEETBEACON,
    UNIT_TYPEID::PROTOSS_FORGE,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_HIGHTEMPLAR,
    UNIT_TYPEID::PROTOSS_IMMORTAL,
    // UNIT_TYPEID::PROTOSS_INTERCEPTOR,
    UNIT_TYPEID::PROTOSS_MOTHERSHIP,
    // UNIT_TYPEID::PROTOSS_MOTHERSHIPCORE,
    UNIT_TYPEID::PROTOSS_NEXUS,
    UNIT_TYPEID::PROTOSS_OBSERVER,
    UNIT_TYPEID::PROTOSS_ORACLE,
    // UNIT_TYPEID::PROTOSS_ORACLESTASISTRAP,
    UNIT_TYPEID::PROTOSS_PHOENIX,
    UNIT_TYPEID::PROTOSS_PHOTONCANNON,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PYLON,
    // UNIT_TYPEID::PROTOSS_PYLONOVERCHARGED,
    UNIT_TYPEID::PROTOSS_ROBOTICSBAY,
    UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
    UNIT_TYPEID::PROTOSS_SENTRY,
    UNIT_TYPEID::PROTOSS_SHIELDBATTERY,
    UNIT_TYPEID::PROTOSS_STALKER,
    UNIT_TYPEID::PROTOSS_STARGATE,
    UNIT_TYPEID::PROTOSS_TEMPEST,
    UNIT_TYPEID::PROTOSS_TEMPLARARCHIVE,
    UNIT_TYPEID::PROTOSS_TWILIGHTCOUNCIL,
    UNIT_TYPEID::PROTOSS_VOIDRAY,
    UNIT_TYPEID::PROTOSS_WARPGATE,
    UNIT_TYPEID::PROTOSS_WARPPRISM,
    // UNIT_TYPEID::PROTOSS_WARPPRISMPHASING,
    UNIT_TYPEID::PROTOSS_ZEALOT,
};

vector<UNIT_TYPEID> unitTypesProtossEconomic = {
    UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
    UNIT_TYPEID::PROTOSS_DARKSHRINE,
    UNIT_TYPEID::PROTOSS_FLEETBEACON,
    UNIT_TYPEID::PROTOSS_FORGE,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_NEXUS,
    UNIT_TYPEID::PROTOSS_NEXUS,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ROBOTICSBAY,
    UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
    UNIT_TYPEID::PROTOSS_STARGATE,
    UNIT_TYPEID::PROTOSS_TEMPLARARCHIVE,
    UNIT_TYPEID::PROTOSS_TWILIGHTCOUNCIL,
    UNIT_TYPEID::PROTOSS_WARPGATE,
};

std::vector<sc2::UNIT_TYPEID> findBestBuildOrderGenetic(const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& startingUnits, const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& target) {
    return findBestBuildOrderGenetic(BuildState(startingUnits), target, nullptr);
}

vector<UNIT_TYPEID> findBestBuildOrderGenetic(const BuildState& startState, const vector<pair<UNIT_TYPEID, int>>& target, const vector<UNIT_TYPEID>* seed) {
    const vector<UNIT_TYPEID>& availableUnitTypes = startState.race == Race::Terran ? unitTypesTerran : (startState.race == Race::Protoss ? unitTypesProtoss : unitTypesZerg);
    const auto& allEconomicUnits = startState.race == Race::Terran ? unitTypesTerranEconomic : (startState.race == Race::Protoss ? unitTypesProtossEconomic : unitTypesZergEconomic);

    // Simulate the starting state until all current events have finished, only then do we know which exact unit types the player will start with.
    // This is important for implicit dependencies in the build order.
    // If say a factory is under construction, we don't want to implictly build another factory if the build order specifies that a tank is supposed to be built.
    BuildState startStateAfterEvents = startState;
    startStateAfterEvents.simulate(startStateAfterEvents.time + 1000000);

    vector<int> startingUnitCounts(availableUnitTypes.size());
    vector<int> startingAddonCountPerUnitType(availableUnitTypes.size());

    for (auto p : startStateAfterEvents.units) {
        startingUnitCounts[indexOf(availableUnitTypes, p.type)] += p.units;
        if (p.addon != UNIT_TYPEID::INVALID) {
            startingUnitCounts[indexOf(availableUnitTypes, getSpecificAddonType(p.type, p.addon))] += p.units;
            startingAddonCountPerUnitType[indexOf(availableUnitTypes, p.type)] += p.units;
        }
    }


    vector<int> actionRequirements(availableUnitTypes.size());
    for (int i = 0; i < actionRequirements.size(); i++) {
        UNIT_TYPEID type = availableUnitTypes[i];
        int count = 0;
        for (auto p : target)
            if (p.first == type || getUnitData(p.first).unit_alias == type)
                count += p.second;
        for (auto p : startStateAfterEvents.units)
            if (p.type == type || getUnitData(p.type).unit_alias == type)
                count -= p.units;
        actionRequirements[i] = max(0, count);
    }

    vector<int> economicUnits;
    for (auto u : allEconomicUnits) {
        for (int i = 0; i < availableUnitTypes.size(); i++)
            if (availableUnitTypes[i] == u)
                economicUnits.push_back(i);
    }

    Stopwatch watch;

    const int POOL_SIZE = 25;
    const float mutationRateAddRemove = 0.05f * 0.5f;
    const float mutationRateMove = 0.05f * 0.5f;
    vector<Gene> generation(POOL_SIZE);
    default_random_engine rnd(time(0));
    for (int i = 0; i < POOL_SIZE; i++) {
        generation[i] = Gene(rnd, actionRequirements);
        generation[i].validate(actionRequirements);
    }
    for (int i = 0; i < 350; i++) {
        if (i == 150 && seed != nullptr) {
            // Add in the seed here
            generation[generation.size()-1] = Gene(*seed, availableUnitTypes, actionRequirements);
            generation[generation.size()-1].validate(actionRequirements);
        }

        vector<float> fitness(generation.size());
        vector<int> indices(generation.size());
        for (int j = 0; j < generation.size(); j++) {
            indices[j] = j;
            fitness[j] = calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, generation[j]);
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
                g.validate(actionRequirements);
                g = locallyOptimizeGene(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, actionRequirements, g);
                g.validate(actionRequirements);
            }

            // Expand build orders
            if (i > 150) {
                for (auto& g : nextGeneration) {
                    // float f1 = calculateFitness(startState, uniqueStartingUnits, availableUnitTypes, g);
                    auto order = g.constructBuildOrder(startState.race, startState.foodAvailable(), startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes);
                    g.buildOrder.clear();
                    for (auto t : order) g.buildOrder.push_back(indexOf(availableUnitTypes, t));
                    // float f2 = calculateFitness(startState, uniqueStartingUnits, availableUnitTypes, g);
                    // if (f1 != f2) {
                    //     cout << "Fitness don't match " << f1 << " " << f2 << endl;
                    // }
                }
            }
        }

        uniform_int_distribution<int> randomParentIndex(0, nextGeneration.size() - 1);
        while (nextGeneration.size() < POOL_SIZE) {
            // nextGeneration.push_back(Gene::crossover(generation[randomParentIndex(rnd)], generation[randomParentIndex(rnd)], rnd, actionRequirements));
            nextGeneration.push_back(generation[randomParentIndex(rnd)]);
        }

        // Note: do not mutate the first gene
        bernoulli_distribution moveMutation(0.5);
        for (int i = 1; i < nextGeneration.size(); i++) {
            nextGeneration[i].mutateMove(mutationRateMove, actionRequirements, rnd);
            nextGeneration[i].mutateAddRemove(mutationRateAddRemove, rnd, actionRequirements, economicUnits);
        }

        swap(generation, nextGeneration);

        // if ((i % 10) == 0) cout << "Best fitness " << fitness[indices[0]] << endl;
        // calculateFitness(startState, uniqueStartingUnits, availableUnitTypes, generation[indices[0]]);
    }

    // cout << "Best fitness " << calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, generation[0]) << endl;
    // printBuildOrder(generation[0].constructBuildOrder(startState.foodAvailable(), startingUnitCounts, availableUnitTypes));
    // printBuildOrderDetailed(startState, generation[0].constructBuildOrder(startState.race, startState.foodAvailable(), startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes));
    return generation[0].constructBuildOrder(startState.race, startState.foodAvailable(), startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes);
}

vector<UNIT_TYPEID> buildOrderProBO = {
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_NEXUS,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
    UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ZEALOT,
    UNIT_TYPEID::PROTOSS_ROBOTICSBAY,
    UNIT_TYPEID::PROTOSS_ZEALOT,
    UNIT_TYPEID::PROTOSS_ZEALOT,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_ZEALOT,
    UNIT_TYPEID::PROTOSS_STALKER,
    UNIT_TYPEID::PROTOSS_STALKER,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ZEALOT,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_STALKER,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ZEALOT 3,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_PYLON,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ZEALOT 3,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_PYLON,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST STALKER,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ADEPT,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST COLLOSUS,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST COLLOSUS,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ADEPT,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ADEPT,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ADEPT,
};

vector<UNIT_TYPEID> buildOrderProBO2 = {
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ADEPT,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ,
    UNIT_TYPEID::PROTOSS_ADEPT ,
    UNIT_TYPEID::PROTOSS_ADEPT ,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ADEPT ,
    UNIT_TYPEID::PROTOSS_ADEPT ,
    UNIT_TYPEID::PROTOSS_ADEPT ,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ,
    // UNIT_TYPEID::PROTOSS_CHRONOBOOST ,
    UNIT_TYPEID::PROTOSS_ADEPT,
};

void unitTestBuildOptimizer() {
    // for (int i = 0; i < unitTypesTerran.size(); i++) {
    //     cout << (int)unitTypesTerran[i] << ": " << i << ",  # " << UnitTypeToName(unitTypesTerran[i]) << endl;
    // }
    // exit(0);
    {
        vector<UNIT_TYPEID> buildOrderTest = {
            UNIT_TYPEID::TERRAN_BANSHEE,
            UNIT_TYPEID::TERRAN_STARPORTTECHLAB,
            UNIT_TYPEID::TERRAN_RAVEN,
        };
        vector<pair<UNIT_TYPEID,int>> startingUnits {
            { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1},
            { UNIT_TYPEID::TERRAN_SCV, 75},
            { UNIT_TYPEID::TERRAN_STARPORT, 1},
        };
        BuildState startState(startingUnits);
        startState.resources.minerals = 50;
        startState.race = Race::Terran;

        Gene gene;
        for (auto b : buildOrderTest) {
            gene.buildOrder.push_back(indexOf(unitTypesTerran, b));
        }

        vector<int> startingUnitCounts(unitTypesTerran.size());
        vector<int> startingAddonCountPerUnitType(unitTypesTerran.size());

        for (auto p : startState.units) {
            startingUnitCounts[indexOf(unitTypesTerran, p.type)] += p.units;
            if (p.addon != UNIT_TYPEID::INVALID) {
                startingUnitCounts[indexOf(unitTypesTerran, getSpecificAddonType(p.type, p.addon))] += p.units;
                startingAddonCountPerUnitType[indexOf(unitTypesTerran, p.type)] += p.units;
            }
        }
        // startState.simulateBuildOrder(gene.constructBuildOrder(startState.race, startState.foodAvailable(), startingUnitCounts, startingAddonCountPerUnitType, unitTypesTerran));
        printBuildOrderDetailed(startState, gene.constructBuildOrder(startState.race, startState.foodAvailable(), startingUnitCounts, startingAddonCountPerUnitType, unitTypesTerran));
    }


    // assert(BuildState({{ UNIT_TYPEID::ZERG_HATCHERY, 1 }, { UNIT_TYPEID::ZERG_DRONE, 12 }}).simulateBuildOrder({ UNIT_TYPEID::ZERG_SPAWNINGPOOL, UNIT_TYPEID::ZERG_ZERGLING }));

    // findBestBuildOrderGenetic({ { UNIT_TYPEID::ZERG_HATCHERY, 1 }, { UNIT_TYPEID::ZERG_DRONE, 12 } }, { { UNIT_TYPEID::ZERG_HATCHERY, 1 }, { UNIT_TYPEID::ZERG_ZERGLING, 12 }, { UNIT_TYPEID::ZERG_MUTALISK, 20 }, { UNIT_TYPEID::ZERG_INFESTOR, 1 } });
    for (int i = 0; i < 0; i++) {
        BuildState startState({ { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } });
        startState.resources.minerals = 50;
        startState.race = Race::Terran;

        findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::TERRAN_VIKINGFIGHTER, 2 }, { UNIT_TYPEID::TERRAN_MEDIVAC, 3 }, { UNIT_TYPEID::TERRAN_BANSHEE, 3 }, { UNIT_TYPEID::TERRAN_MARINE, 20 }, { UNIT_TYPEID::TERRAN_BUNKER, 1 } });
        // findBestBuildOrderGenetic({ { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_BARRACKSREACTOR, 0 }, { UNIT_TYPEID::TERRAN_MARINE, 30 }, { UNIT_TYPEID::TERRAN_MARAUDER, 0 } });

        if(false) {
            BuildState startState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 } });
            startState.resources.minerals = 50;
            startState.race = Race::Protoss;
            // findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::PROTOSS_ADEPT, 23 } });
            // findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::PROTOSS_ZEALOT, 20 }, { UNIT_TYPEID::PROTOSS_STALKER, 30 }, { UNIT_TYPEID::PROTOSS_ADEPT, 12 }, { UNIT_TYPEID::PROTOSS_COLOSSUS, 2 } });
            findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::PROTOSS_ZEALOT, 5 }, { UNIT_TYPEID::PROTOSS_STALKER, 3 }, { UNIT_TYPEID::PROTOSS_ADEPT, 12 }, { UNIT_TYPEID::PROTOSS_COLOSSUS, 2 } });
            printBuildOrderDetailed(startState, buildOrderProBO);
        }
    }

    
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 1 } });
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 2 } });
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 5 } });
    // logBuildOrder(optimizer.calculate_build_order();
    // logBuildOrder(optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_MARINE, 5 } }));
}
