#include "BuildOptimizerGenetic.h"
#include <algorithm>
#include <limits>
#include <map>
#include <random>
#include <stack>
#include "utilities/mappings.h"
#include "utilities/predicates.h"
#include "utilities/profiler.h"
#include "utilities/stdutils.h"

using namespace std;
using namespace sc2;

using GeneUnitType = int;
const BuildOrderFitness BuildOrderFitness::ReallyBad = { 100000, BuildResources(0,0), { 0, 0 } };

void printBuildOrder(vector<UNIT_TYPEID> buildOrder);

BuildState::BuildState(const ObservationInterface* observation, Unit::Alliance alliance, Race race, BuildResources resources, float time) : time(time), race(race), resources(resources) {
    map<UNIT_TYPEID, int> startingUnitsCount;
    map<UNIT_TYPEID, int> targetUnitsCount;

    auto ourUnits = observation->GetUnits(alliance);

    for (int i = 0; i < ourUnits.size(); i++) {
        auto unitType = ourUnits[i]->unit_type;

        // Addons are handled when the unit they are attached to are handled
        if (isAddon(unitType)) continue;

        if (ourUnits[i]->build_progress < 1) {
            if (race == Race::Terran) {
                // Ignore (will be handled by the order code below)
            } else {
                unitType = canonicalize(unitType);
                float buildTime = ticksToSeconds(getUnitData(unitType).build_time);
                float remainingTime = (1 - ourUnits[i]->build_progress) * buildTime;
                auto event = BuildEvent(BuildEventType::FinishedUnit, time + remainingTime, UNIT_TYPEID::INVALID, getUnitData(unitType).ability_id);
                // Note: don't bother to handle addons as this code never runs for terran
                addEvent(event);
            }
        } else {
            if (ourUnits[i]->add_on_tag != NullTag) {
                auto* addon = observation->GetUnit(ourUnits[i]->add_on_tag);
                // Simplify to e.g. TERRAN_REACTOR instead of TERRAN_BARRACKSREACTOR
                auto addonType = simplifyUnitType(addon->unit_type);
                assert(addonType != UNIT_TYPEID::INVALID);
                addUnits(canonicalize(unitType), addonType, 1);
            } else {
                addUnits(canonicalize(unitType), 1);
            }
        }

        // Note: at least during replays, orders don't seem to contain the target_unit_tag for build construction
        
        for (auto order : ourUnits[i]->orders) {
            auto createdUnit = abilityToUnit(order.ability_id);
            if (createdUnit != UNIT_TYPEID::INVALID) {
                if (canonicalize(createdUnit) == canonicalize(unitType)) {
                    // This is just a morph ability (e.g. lower a supply depot)
                    break;
                }

                float buildProgress = order.progress;
                if (order.target_unit_tag != NullTag) {
                    auto* targetUnit = observation->GetUnit(order.target_unit_tag);
                    if (targetUnit == nullptr) {
                        cerr << "Target for order does not seem to exist!??" << endl;
                        break;
                    }

                    assert(targetUnit != nullptr);

                    // In some cases the target unit can be something else, for example when constructing a refinery the target unit is the vespene geyser for a while
                    if (targetUnit->owner == ourUnits[i]->owner) {
                        buildProgress = targetUnit->build_progress;
                        // TODO: build_progress is not empirically always order.progress, when is it not so?
                        assert(targetUnit->build_progress >= 0 && targetUnit->build_progress <= 1);
                    }
                }

                float buildTime = ticksToSeconds(getUnitData(createdUnit).build_time);
                // TODO: Is this linear?
                float remainingTime = (1 - buildProgress) * buildTime;
                auto event = BuildEvent(BuildEventType::FinishedUnit, time + remainingTime, canonicalize(unitType), order.ability_id);
                if (ourUnits[i]->add_on_tag != NullTag) {
                    auto* addon = observation->GetUnit(ourUnits[i]->add_on_tag);
                    assert(addon != nullptr);
                    // Normalize from e.g. TERRAN_BARRACKSTECHLAB to TERRAN_TECHLAB
                    event.casterAddon = simplifyUnitType(addon->unit_type);
                }

                // Make the caster busy first
                makeUnitsBusy(event.caster, event.casterAddon, 1);
                addEvent(event);
            }

            // Only process the first order (this bot should never have more than one anyway)
            break;
        }
    }

    vector<Point2D> basePositions;
    for (auto u : ourUnits) {
        if (isTownHall(u->unit_type)) {
            basePositions.push_back(u->pos);
            baseInfos.push_back(BaseInfo(0, 0, 0));
        }
    }
    auto neutralUnits = observation->GetUnits(Unit::Alliance::Neutral);
    for (auto u : neutralUnits) {
        if (u->mineral_contents > 0) {
            for (int i = 0; i < baseInfos.size(); i++) {
                if (DistanceSquared2D(u->pos, basePositions[i]) < 10*10) {
                    baseInfos[i].remainingMinerals += u->mineral_contents;
                    break;
                }
            }
        }
    }
}

void BuildState::makeUnitsBusy(UNIT_TYPEID type, UNIT_TYPEID addon, int delta) {
    if (delta == 0)
        return;

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
    if (delta == 0)
        return;

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

void MiningSpeed::simulateMining (BuildState& state, float dt) const {
    float totalWeight = 0;
    for (auto& base : state.baseInfos) {
        auto slots = base.mineralSlots();
        float weight = slots.first * 1.5f + slots.second;
        totalWeight += weight;
    }
    float normalizationFactor = 1.0f / (totalWeight + 0.0001f);
    float deltaMineralsPerWeight = mineralsPerSecond * dt * normalizationFactor;
    for (auto& base : state.baseInfos) {
        auto slots = base.mineralSlots();
        float weight = slots.first * 1.5f + slots.second;
        base.mineMinerals(deltaMineralsPerWeight * weight);
    }
    state.resources.minerals += mineralsPerSecond * dt;
    state.resources.vespene += vespenePerSecond * dt;
}

MiningSpeed BuildState::miningSpeed() const {
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

    int highYieldMineralHarvestingSlots = 0;
    int lowYieldMineralHarvestingSlots = 0;
    for (int i = 0; i < bases; i++) {
        if (i < baseInfos.size()) {
            auto t = baseInfos[i].mineralSlots();
            highYieldMineralHarvestingSlots += t.first;
            lowYieldMineralHarvestingSlots += t.second;
        } else {
            // Assume lots of minerals
            highYieldMineralHarvestingSlots += 16;
            lowYieldMineralHarvestingSlots += 8;
        }
    }

    int vespeneMining = min(harvesters / 2, geysers * 3);
    int mineralMining = harvesters - vespeneMining;

    // Maximum effective harvesters (todo: account for more things)
    // First 2 harvesters per mineral field yield more minerals than the 3rd one.
    int highYieldHarvesters = min(highYieldMineralHarvestingSlots, mineralMining);
    int lowYieldHarvesters = min(lowYieldMineralHarvestingSlots, mineralMining - highYieldHarvesters);

    // TODO: Check units here!
    const float FasterSpeedMultiplier = 1.4f;
    const float LowYieldMineralsPerMinute = 22 * FasterSpeedMultiplier;
    const float HighYieldMineralsPerMinute = 40 * FasterSpeedMultiplier;
    const float VespenePerMinute = 38 * FasterSpeedMultiplier;
    const float MinutesPerSecond = 1 / 60.0f;

    MiningSpeed speed;
    speed.mineralsPerSecond = (lowYieldHarvesters * LowYieldMineralsPerMinute + highYieldHarvesters * HighYieldMineralsPerMinute) * MinutesPerSecond;
    speed.vespenePerSecond = vespeneMining * VespenePerMinute * MinutesPerSecond;

    return speed;
}

float BuildState::timeToGetResources(MiningSpeed miningSpeed, float mineralCost, float vespeneCost) const {
    mineralCost -= resources.minerals;
    vespeneCost -= resources.vespene;
    float time = 0;
    if (mineralCost > 0) {
        if (miningSpeed.mineralsPerSecond == 0)
            return numeric_limits<float>::infinity();
        time = mineralCost / miningSpeed.mineralsPerSecond;
    }
    if (vespeneCost > 0) {
        if (miningSpeed.vespenePerSecond == 0)
            return numeric_limits<float>::infinity();
        time = max(time, vespeneCost / miningSpeed.vespenePerSecond);
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
        currentMiningSpeed.simulateMining(*this, dt);
        time = ev.time;

        ev.apply(*this);

        if (ev.impactsEconomy()) {
            currentMiningSpeed = miningSpeed();
        } else {
            // Ideally this would always hold, but when we simulate actual bases with mineral patches the approximations used are not entirely accurate and the mining rate may change even when there is no economically significant event
            assert(baseInfos.size() > 0 || currentMiningSpeed == miningSpeed());
        }
    }

    events.erase(events.begin(), events.begin() + eventIndex);

    {
        float dt = endTime - time;
        currentMiningSpeed.simulateMining(*this, dt);
        time = endTime;
    }
}

bool BuildState::simulateBuildOrder(vector<UNIT_TYPEID> buildOrder, function<void(int)> callback, bool waitUntilItemsFinished) {
    float lastEventInBuildOrder = 0;

    // Loop through the build order
    int buildIndex = -1;
    for (auto unitType : buildOrder) {
        buildIndex++;
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
                    cout << "No tech at index " << buildIndex << endl;
                    return false;
                    cout << "Requires " << UnitTypeToName(unitData.tech_requirement) << endl;
                    cout << foodAvailable() << " " << unitData.food_required << endl;
                    cout << UnitTypeToName(unitType) << endl;
                    printBuildOrder(buildOrder);
                    cout << "Current unit counts:" << endl;
                    for (auto u : units) {
                        cout << UnitTypeToName(u.type) << " " << UnitTypeToName(u.addon) << " " << u.units << endl;
                    }
                    // __builtin_trap();
                    // exit(1);
                    return false;
                }

                simulate(events[0].time);
                continue;
            }

            // TODO: Handles food?

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
            // When do we have enough resources for this item
            float eventTime = time + timeToGetResources(currentMiningSpeed, mineralCost, vespeneCost);

            // If it would be after the next economically significant event then the time estimate is likely not accurate (the mining speed might change in the middle)
            if (eventTime > nextSignificantEvent) {
                simulate(nextSignificantEvent);
                continue;
            }

            ABILITY_ID ability = unitData.ability_id;
            // Make sure that some unit can cast this ability
            assert(abilityToCasterUnit(ability).size() > 0);

            // Find an appropriate caster for this ability
            BuildUnitInfo* casterUnit = nullptr;
            UNIT_TYPEID casterUnitType = UNIT_TYPEID::INVALID;
            UNIT_TYPEID casterAddonType = UNIT_TYPEID::INVALID;
            for (UNIT_TYPEID caster : abilityToCasterUnit(ability)) {
                for (auto& casterCandidate : units) {
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

            // If we don't have a caster yet, then we might have to simulate a bit (the caster might be training right now, or currently busy)
            if (casterUnit == nullptr) {
                if (casterUnitType == UNIT_TYPEID::ZERG_LARVA) {
                    addUnits(UNIT_TYPEID::ZERG_LARVA, 1);
                    continue;
                }

                if (events.empty()) {
                    printBuildOrder(buildOrder);
                    cout << "No possible caster " << UnitTypeToName(unitType) << endl;
                    for (auto& casterCandidate : units) {
                        cout << "Caster: " << UnitTypeToName(casterCandidate.type) << " " << casterCandidate.units << "/" << casterCandidate.availableUnits() << " " << UnitTypeToName(casterCandidate.addon) << endl;
                    }
                    // exit(1);
                    return false;
                }

                simulate(events[0].time);
                continue;
            }

            if (isinf(eventTime)) {
                // This can happen in some cases.
                // Most common is when the unit requires vespene gas, but the player only has 1 scv and that one will be allocated to minerals.
                return false;
            }

            // Fast forward until we can pay for the item
            simulate(eventTime);

            // The simulation may invalidate pointers, so find the caster again
            for (auto& casterCandidate : units) {
                if (casterCandidate.type == casterUnitType && casterCandidate.addon == casterAddonType && casterCandidate.availableUnits() > 0) {
                    casterUnit = &casterCandidate;
                }
            }
            assert(casterUnit != nullptr);

            // Pay for the item
            resources.minerals -= mineralCost;
            resources.vespene -= vespeneCost;

            // Mark the caster as being busy
            casterUnit->busyUnits++;
            assert(casterUnit->availableUnits() >= 0);

            float buildTime = ticksToSeconds(unitData.build_time);

            // Compensate for workers having to move to the building location
            if (isStructure(unitType)) {
                buildTime += 4;
            }

            // Create a new event for when the item is complete
            auto newEvent = BuildEvent(BuildEventType::FinishedUnit, time + buildTime, casterUnit->type, ability);
            newEvent.casterAddon = casterUnit->addon;
            lastEventInBuildOrder = max(lastEventInBuildOrder, newEvent.time);
            addEvent(newEvent);
            if (casterUnit->type == UNIT_TYPEID::PROTOSS_PROBE) {
                addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, time + 4, UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::INVALID));
            }

            if (callback != nullptr)
                callback(buildIndex);
            break;
        }
    }

    if (waitUntilItemsFinished) simulate(lastEventInBuildOrder);
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

void BuildEvent::apply(BuildState& state) const {
    switch (type) {
        case FinishedUnit: {
            UNIT_TYPEID unit = abilityToUnit(ability);

            // Probes are special because they don't actually have to stay while the building is being built
            // Another event will make them available a few seconds after the order has been issued.
            if (caster != UNIT_TYPEID::PROTOSS_PROBE && caster != UNIT_TYPEID::INVALID) {
                state.makeUnitsBusy(caster, casterAddon, -1);
            }

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
        case MakeUnitAvailable: {
            state.makeUnitsBusy(caster, casterAddon, -1);
            break;
        }
    }
}

/** Adds all dependencies of the required type to the requirements stack in the order that they need to be performed in order to fulfill all preconditions for building/training the required type
 * 
 * For example if the player only has some SCVs and a command center and the required type is a marine, then both a barracks and a supply depot will be added to the stack.
 * Only takes care of direct tech dependencies, not indirect ones like supply or resource requirements.
 */
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

/** Finalizes the gene's build order by adding in all implicit steps */
vector<UNIT_TYPEID> addImplicitBuildOrderSteps(const vector<int>& buildOrder, Race race, float startingFood, const vector<int>& startingUnitCounts, const vector<int>& startingAddonCountPerUnitType, const vector<UNIT_TYPEID>& availableUnitTypes, vector<bool>* outIsOriginalItem = nullptr) {
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
            // The last item that we add is the one from the original build order
            if (outIsOriginalItem != nullptr) outIsOriginalItem->push_back(reqs.empty());
        }
    }

    return finalBuildOrder;
}

/** A gene represents a build order.
 * The build order may contain many implicit steps which will be added when the build order is finalized.
 * For example if any build item has any preconditions (e.g. training a marine requires a barracks which requires a supply depot, etc.) then when the build order is finalized any
 * implicit steps are added such that the build order can always be performed almost no matter what the build order is.
 * That is, the simulation will not return an error because for example it tried to train a marine even though the player has no baracks.
 * 
 * The build order is represented as a vector of integers which are indices into a provided availableUnitTypes list (for example all relevant unit types for a terran player).
 */
struct Gene {
    // Indices are into the availableUnitTypes list
    vector<GeneUnitType> buildOrder;

    /** Validates that the build order will train/build the given units and panics otherwise */
    void validate(const vector<int>& actionRequirements) const {
#if DEBUG
        vector<int> remainingRequirements = actionRequirements;
        for (auto type : buildOrder)
            remainingRequirements[type]--;
        for (auto r : remainingRequirements)
            assert(r <= 0);
#endif
    }

    /** Mutates the build order by moving items around randomly.
     * Ensures that the build order will still create the units given by the action requirements.
     * 
     * The action requirements is a list as long as availableUnitTypes that specifies for each unit type how many that the build order should train/build.
     */
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

    /** Mutates the build order by adding or removing items.
     * Ensures that the build order will still create the units given by the action requirements.
     * 
     * The action requirements is a list as long as availableUnitTypes that specifies for each unit type how many that the build order should train/build.
     */
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

    /** Creates a random build order that will train/build the given units.
     * 
     * The action requirements is a list as long as availableUnitTypes that specifies for each unit type how many that the build order should train/build.
     */
    Gene(default_random_engine& seed, const vector<int>& actionRequirements) {
        for (GeneUnitType i = 0; i < actionRequirements.size(); i++) {
            for (int j = actionRequirements[i] - 1; j >= 0; j--)
                buildOrder.push_back(i);
        }
        shuffle(buildOrder.begin(), buildOrder.end(), seed);
    }

    /** Creates a gene from a given build order */
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
    
    vector<UNIT_TYPEID> constructBuildOrder(Race race, float startingFood, const vector<int>& startingUnitCounts, const vector<int>& startingAddonCountPerUnitType, const vector<UNIT_TYPEID>& availableUnitTypes) const {
        return addImplicitBuildOrderSteps(buildOrder, race, startingFood, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes);
    }
};

void printBuildOrderDetailed(const BuildState& startState, vector<UNIT_TYPEID> buildOrder, const vector<bool>* highlight) {
    BuildState state = startState;
    cout << "Starting units" << endl;
    for (auto u : startState.units) {
        cout << "\t" << u.units << "x " << UnitTypeToName(u.type);
        if (u.addon != UNIT_TYPEID::INVALID)
            cout << " + " << UnitTypeToName(u.addon);
        cout << endl;
    }
    cout << "Build order size " << buildOrder.size() << endl;
    bool success = state.simulateBuildOrder(buildOrder, [&](int i) {
        if (highlight != nullptr && (*highlight)[i]) {
            // Color the text
            cout << "\x1b[" << 48 << ";2;" << 228 << ";" << 26 << ";" << 28 << "m"; 
        }
        cout << "Step " << i << "\t" << (int)(state.time / 60.0f) << ":" << (int)(fmod(state.time, 60.0f)) << "\t" << UnitTypeToName(buildOrder[i]) << " "
             << "food: " << state.foodAvailable() << " resources: " << (int)state.resources.minerals << "+" << (int)state.resources.vespene << " " << (state.baseInfos.size() > 0 ? state.baseInfos[0].remainingMinerals : 0);

        // Reset color
        cout << "\033[0m";
        cout << endl;
    });

    cout << (success ? "Finished at " : "Failed at ");
    cout << (int)(state.time / 60.0f) << ":" << (int)(fmod(state.time, 60.0f)) << " resources: " << state.resources.minerals << "+" << state.resources.vespene << " mining speed: " << (int)round(state.miningSpeed().mineralsPerSecond*60) << "/min + " << (int)round(state.miningSpeed().vespenePerSecond*60) << "/min" << endl;
}

void printBuildOrder(vector<UNIT_TYPEID> buildOrder) {
    cout << "Build order size " << buildOrder.size() << endl;
    for (int i = 0; i < buildOrder.size(); i++) {
        cout << "Step " << i << " " << UnitTypeToName(buildOrder[i]) << endl;
    }
}

float BuildOrderFitness::score() const {
    float s = -fmax(time, 2 * 60.0f);
    s += ((resources.minerals + 2 * resources.vespene) + (miningSpeed.mineralsPerSecond + 2 * miningSpeed.vespenePerSecond) * 60) * 0.001f;
    // s = log(s) - time/400.0f;
    return s;
}

/** Calculates the fitness of a given build order gene, a higher value is better */
BuildOrderFitness calculateFitness(const BuildState& startState, const vector<int>& startingUnitCounts, const vector<int>& startingAddonCountPerUnitType, const vector<UNIT_TYPEID>& availableUnitTypes, const Gene& gene) {
    BuildState state = startState;
    vector<float> finishedTimes;
    auto buildOrder = gene.constructBuildOrder(startState.race, startState.foodAvailable(), startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes);
    if (!state.simulateBuildOrder(buildOrder, [&](int index) {
            finishedTimes.push_back(state.time);
        })) {
        // Build order could not be executed, that is bad.
        return BuildOrderFitness::ReallyBad;
    }

    // Find the average completion time of the army units in the build order (they are usually the important parts, though non army units also contribute a little bit)
    float avgTime = state.time * 0.00001f;
    float totalWeight = 0.00001f;
    for (int i = 0; i < finishedTimes.size(); i++) {
        float t = finishedTimes[i];
        float w = isArmy(buildOrder[i]) ? 1.0f : 0.1f;
        totalWeight += w;
        avgTime += w * (t + 20);  // +20 to take into account that we want the finished time of the unit, but we only have the start time
    }

    avgTime /= totalWeight;

    // Simulate until at least the 2 minutes mark, this ensures that the agent will do some economic stuff if nothing else
    state.simulate(60 * 2);

    auto miningSpeed = state.miningSpeed();
    return BuildOrderFitness(avgTime * 2, state.resources, miningSpeed);
    // return -max(avgTime * 2, 2 * 60.0f) + (state.resources.minerals + 2 * state.resources.vespene) * 0.001 + (miningSpeed.mineralsPerSecond + 2 * miningSpeed.vespenePerSecond) * 60 * 0.005;
}

/** Try really hard to do optimize the gene.
 * This will try to swap adjacent items in the build order as well as trying to remove all non-essential items.
 */
// TODO: Add operation to remove all items that are implied anyway (i.e. if removing the item and then adding in implicit steps returns the same result as just adding in the implicit steps)
Gene locallyOptimizeGene(const BuildState& startState, const vector<int>& startingUnitCounts, const vector<int>& startingAddonCountPerUnitType, const vector<UNIT_TYPEID>& availableUnitTypes, const vector<int>& actionRequirements, const Gene& gene) {
    vector<int> currentActionRequirements = actionRequirements;
    for (auto b : gene.buildOrder)
        currentActionRequirements[b]--;

    auto startFitness = calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, gene);
    auto fitness = startFitness;
    Gene newGene = gene;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < newGene.buildOrder.size(); j++) {
            if (j == newGene.buildOrder.size() - 1 || newGene.buildOrder[j] != newGene.buildOrder[j + 1]) {
                // Check if the item is non-essential
                if (currentActionRequirements[newGene.buildOrder[j]] < 0) {
                    // Try removing
                    auto orig = newGene.buildOrder[j];
                    newGene.buildOrder.erase(newGene.buildOrder.begin() + j);
                    auto newFitness = calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, newGene);
                    if (fitness < newFitness) {
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
                if (j + 1 < newGene.buildOrder.size()) {
                    swap(newGene.buildOrder[j], newGene.buildOrder[j + 1]);
                    auto newFitness = calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, newGene);

                    if (fitness < newFitness) {
                        fitness = newFitness;
                    } else {
                        // Revert swap
                        swap(newGene.buildOrder[j], newGene.buildOrder[j + 1]);
                    }
                }
            }
        }
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

pair<vector<int>, vector<int>> calculateStartingUnitCounts(const BuildState& startState, const vector<UNIT_TYPEID>& availableUnitTypes) {
    vector<int> startingUnitCounts(availableUnitTypes.size());
    vector<int> startingAddonCountPerUnitType(availableUnitTypes.size());

    for (auto p : startState.units) {
        startingUnitCounts[indexOf(availableUnitTypes, p.type)] += p.units;
        if (p.addon != UNIT_TYPEID::INVALID) {
            startingUnitCounts[indexOf(availableUnitTypes, getSpecificAddonType(p.type, p.addon))] += p.units;
            startingAddonCountPerUnitType[indexOf(availableUnitTypes, p.type)] += p.units;
        }
    }
    return { startingUnitCounts, startingAddonCountPerUnitType };
}

const vector<UNIT_TYPEID>& getAvailableUnitTypesForRace (Race race) {
    return race == Race::Terran ? unitTypesTerran : (race == Race::Protoss ? unitTypesProtoss : unitTypesZerg);
}

const vector<UNIT_TYPEID>& getEconomicUnitTypesForRace (Race race) {
    return race == Race::Terran ? unitTypesTerranEconomic : (race == Race::Protoss ? unitTypesProtossEconomic : unitTypesZergEconomic);
}

pair<vector<UNIT_TYPEID>, vector<bool>> expandBuildOrderWithImplicitSteps (const BuildState& startState, vector<UNIT_TYPEID> buildOrder) {
    const vector<UNIT_TYPEID>& availableUnitTypes = getAvailableUnitTypesForRace(startState.race);

    // Simulate the starting state until all current events have finished, only then do we know which exact unit types the player will start with.
    // This is important for implicit dependencies in the build order.
    // If say a factory is under construction, we don't want to implictly build another factory if the build order specifies that a tank is supposed to be built.
    BuildState startStateAfterEvents = startState;
    startStateAfterEvents.simulate(startStateAfterEvents.time + 1000000);

    vector<int> startingUnitCounts;
    vector<int> startingAddonCountPerUnitType;
    tie(startingUnitCounts, startingAddonCountPerUnitType) = calculateStartingUnitCounts(startStateAfterEvents, availableUnitTypes);

    vector<int> remappedBuildOrder(buildOrder.size());
    for (int i = 0; i < buildOrder.size(); i++) remappedBuildOrder[i] = indexOf(availableUnitTypes, buildOrder[i]);

    vector<bool> partOfOriginalBuildOrder;
    vector<UNIT_TYPEID> finalBuildOrder = addImplicitBuildOrderSteps(remappedBuildOrder, startStateAfterEvents.race, startStateAfterEvents.foodAvailable(), startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, &partOfOriginalBuildOrder);
    return { finalBuildOrder, partOfOriginalBuildOrder };
}

/** Finds the best build order using an evolutionary algorithm */
vector<UNIT_TYPEID> findBestBuildOrderGenetic(const BuildState& startState, const vector<pair<UNIT_TYPEID, int>>& target, const vector<UNIT_TYPEID>* seed) {
    const vector<UNIT_TYPEID>& availableUnitTypes = getAvailableUnitTypesForRace(startState.race);
    const vector<UNIT_TYPEID>& allEconomicUnits = getEconomicUnitTypesForRace(startState.race);

    // Simulate the starting state until all current events have finished, only then do we know which exact unit types the player will start with.
    // This is important for implicit dependencies in the build order.
    // If say a factory is under construction, we don't want to implictly build another factory if the build order specifies that a tank is supposed to be built.
    BuildState startStateAfterEvents = startState;
    startStateAfterEvents.simulate(startStateAfterEvents.time + 1000000);

    vector<int> startingUnitCounts;
    vector<int> startingAddonCountPerUnitType;
    tie(startingUnitCounts, startingAddonCountPerUnitType) = calculateStartingUnitCounts(startStateAfterEvents, availableUnitTypes);

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

    // const int POOL_SIZE = 25;
    const int POOL_SIZE = 25;
    const float mutationRateAddRemove = 0.025f;
    const float mutationRateMove = 0.025f;
    vector<Gene> generation(POOL_SIZE);
    default_random_engine rnd(time(0));
    for (int i = 0; i < POOL_SIZE; i++) {
        generation[i] = Gene(rnd, actionRequirements);
        generation[i].validate(actionRequirements);
    }
    for (int i = 0; i <= 350; i++) {
        if (i == 150 && seed != nullptr) {
            // Add in the seed here
            generation[generation.size() - 1] = Gene(*seed, availableUnitTypes, actionRequirements);
            generation[generation.size() - 1].validate(actionRequirements);
        }

        vector<BuildOrderFitness> fitness(generation.size());
        vector<int> indices(generation.size());
        for (int j = 0; j < generation.size(); j++) {
            indices[j] = j;
            fitness[j] = calculateFitness(startState, startingUnitCounts, startingAddonCountPerUnitType, availableUnitTypes, generation[j]);
        }

        sortByValueDescending<int, BuildOrderFitness>(indices, [=](int index) { return fitness[index]; });
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
                    for (auto t : order)
                        g.buildOrder.push_back(indexOf(availableUnitTypes, t));
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
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_PYLON,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ADEPT,
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
        vector<pair<UNIT_TYPEID, int>> startingUnits{
            { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 },
            { UNIT_TYPEID::TERRAN_SCV, 75 },
            { UNIT_TYPEID::TERRAN_STARPORT, 1 },
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
    for (int i = 0; i < 1; i++) {
        BuildState startState({ { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } });
        startState.resources.minerals = 50;
        startState.race = Race::Terran;

        auto bo = findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::TERRAN_MARAUDER, 12 }, { UNIT_TYPEID::TERRAN_BANSHEE, 0 } });
        printBuildOrderDetailed(startState, bo);
        exit(0);

        // findBestBuildOrderGenetic({ { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_BARRACKSREACTOR, 0 }, { UNIT_TYPEID::TERRAN_MARINE, 30 }, { UNIT_TYPEID::TERRAN_MARAUDER, 0 } });

        if (true) {
            BuildState startState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 } });
            startState.resources.minerals = 50;
            startState.race = Race::Protoss;
            // Initial delay before harvesters start mining properly
            startState.makeUnitsBusy(UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::INVALID, 12);
            for (int i = 0; i < 12; i++) startState.addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, 4, UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::INVALID));

            startState.baseInfos = { BaseInfo(10800, 1000, 1000) };

            vector<UNIT_TYPEID> bo5 = {
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_OBSERVER,
                UNIT_TYPEID::PROTOSS_IMMORTAL,
                UNIT_TYPEID::PROTOSS_CARRIER,
            };

            vector<UNIT_TYPEID> bo6 = {
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_OBSERVER,
                UNIT_TYPEID::PROTOSS_CARRIER,
                UNIT_TYPEID::PROTOSS_IMMORTAL,
            };

            // findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::PROTOSS_ADEPT, 23 } });
            // findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::PROTOSS_ZEALOT, 20 }, { UNIT_TYPEID::PROTOSS_STALKER, 30 }, { UNIT_TYPEID::PROTOSS_ADEPT, 12 }, { UNIT_TYPEID::PROTOSS_COLOSSUS, 2 } });
            // auto bo = findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::PROTOSS_PHOENIX, 3 }, { UNIT_TYPEID::PROTOSS_ZEALOT, 15 }, { UNIT_TYPEID::PROTOSS_CARRIER, 1 }, { UNIT_TYPEID::PROTOSS_OBSERVER, 1 }, { UNIT_TYPEID::PROTOSS_IMMORTAL, 1 } });
            auto bo = findBestBuildOrderGenetic(startState, { { UNIT_TYPEID::PROTOSS_ZEALOT, 60 }, { UNIT_TYPEID::PROTOSS_PHOENIX, 5 }, { UNIT_TYPEID::PROTOSS_CARRIER, 1 }, { UNIT_TYPEID::PROTOSS_IMMORTAL, 1 } });

            /*vector<UNIT_TYPEID> bo = {
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_OBSERVER,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_NEXUS,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_IMMORTAL,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_CARRIER,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
            };*/

            vector<UNIT_TYPEID> bo2 = {
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_OBSERVER,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_IMMORTAL,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_CARRIER,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
            };

            vector<UNIT_TYPEID> bo4 = {
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_OBSERVER,
                UNIT_TYPEID::PROTOSS_CARRIER,
                UNIT_TYPEID::PROTOSS_IMMORTAL,
            };

            vector<UNIT_TYPEID> bo7 = {
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_OBSERVER,
                UNIT_TYPEID::PROTOSS_CARRIER,
                UNIT_TYPEID::PROTOSS_IMMORTAL,
            };

            vector<UNIT_TYPEID> bo8 = {
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_NEXUS,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PROBE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ASSIMILATOR,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PYLON,
                UNIT_TYPEID::PROTOSS_FLEETBEACON,
                UNIT_TYPEID::PROTOSS_ZEALOT,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_NEXUS,
                UNIT_TYPEID::PROTOSS_CARRIER,
                UNIT_TYPEID::PROTOSS_STARGATE,
                UNIT_TYPEID::PROTOSS_GATEWAY,
                UNIT_TYPEID::PROTOSS_PHOENIX,
                UNIT_TYPEID::PROTOSS_IMMORTAL,
                UNIT_TYPEID::PROTOSS_PHOENIX,
            };

            printBuildOrderDetailed(startState, bo);
            printBuildOrderDetailed(startState, bo4);
            printBuildOrderDetailed(startState, bo5);
            printBuildOrderDetailed(startState, bo6);
            printBuildOrderDetailed(startState, bo7);
            printBuildOrderDetailed(startState, bo8);

            /*for (int i = 0; i < bo2.size(); i++) {
                auto bo3 = bo2;
                bo3.insert(bo3.begin() + i, UNIT_TYPEID::PROTOSS_PROBE);
                auto state2 = startState;
                bool s = state2.simulateBuildOrder(bo3);
                if (!s) {
                    bo3.insert(bo3.begin() + i, UNIT_TYPEID::PROTOSS_PYLON);
                    state2 = startState;
                    s = state2.simulateBuildOrder(bo3);
                }
                if (!s) continue;
                BuildOrderFitness f = BuildOrderFitness(state2.time, state2.resources, state2.miningSpeed());
                cout << "(" << i << ", " << state2.time << "," << f.score() << ")," << endl;
            }
            printBuildOrderDetailed(startState, bo);
            printBuildOrderDetailed(startState, bo2);*/
            // printBuildOrderDetailed(startState, buildOrderProBO);
        }
    }

    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 1 } });
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 2 } });
    // optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 1 } }, { { UNIT_TYPEID::TERRAN_SCV, 5 } });
    // logBuildOrder(optimizer.calculate_build_order();
    // logBuildOrder(optimizer.calculate_build_order(Race::Terran, { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } }, { { UNIT_TYPEID::TERRAN_MARINE, 5 } }));
}
