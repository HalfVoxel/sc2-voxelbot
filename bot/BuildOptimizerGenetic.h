#pragma once
#include <vector>
#include <cmath>
#include "sc2api/sc2_interfaces.h"

struct BuildState;

struct BuildResources {
    float minerals;
    float vespene;

    BuildResources(float minerals, float vespene)
        : minerals(minerals), vespene(vespene) {}
};

struct MiningSpeed {
    float mineralsPerSecond;
    float vespenePerSecond;

    /** Simulate mining with the given speed for the given amount of time */
    void simulateMining (BuildState& state, float dt) const;

    inline bool operator== (const MiningSpeed& other) const {
        return mineralsPerSecond == other.mineralsPerSecond && vespenePerSecond == other.vespenePerSecond;
    }
};

const std::vector<sc2::UNIT_TYPEID>& getAvailableUnitTypesForRace (sc2::Race race);
const std::vector<sc2::UNIT_TYPEID>& getEconomicUnitTypesForRace (sc2::Race race);

struct BuildUnitInfo {
    /** Type of the unit */
    sc2::UNIT_TYPEID type;

    /** Optional addon attached to the unit (or UNIT_TYPEID::INVALID otherwise) */
    sc2::UNIT_TYPEID addon;

    /** Total number of units */
    int units;

    /** How many units out of the total that are busy right now.
     * E.g. with constructing a building, training a unit, etc.
     * Technically this is the number of casting slots for abilities rather than the number of units,
     * in case the building has a reactor the number of busy units can be up to two times the number of units because each unit can train two units at the same time.
     */
    int busyUnits;

    BuildUnitInfo()
        : type(sc2::UNIT_TYPEID::INVALID), addon(sc2::UNIT_TYPEID::INVALID), units(0), busyUnits(0) {}
    BuildUnitInfo(sc2::UNIT_TYPEID type, sc2::UNIT_TYPEID addon, int units)
        : type(type), addon(addon), units(units), busyUnits(0) {}

    /** Number of units that are available for casting abilities */
    inline int availableUnits() const {
        if (addon == sc2::UNIT_TYPEID::TERRAN_REACTOR) {
            return units - busyUnits / 2;
        } else {
            return units - busyUnits;
        }
    }
};

enum BuildEventType {
    FinishedUnit,
    SpawnLarva,
    MuleTimeout,
    MakeUnitAvailable,  // Un-busy unit
};

struct BuildEvent {
    /** Type of event */
    BuildEventType type;
    /** The ability that is being cast */
    sc2::ABILITY_ID ability;
    /** Unit type of the caster of the ability */
    sc2::UNIT_TYPEID caster;
    /** Addon unit type of the caster of the ability (if any) */
    sc2::UNIT_TYPEID casterAddon;
    /** Time at which this event will happen */
    float time;

    BuildEvent(BuildEventType type, float time, sc2::UNIT_TYPEID caster, sc2::ABILITY_ID ability)
        : type(type), ability(ability), caster(caster), casterAddon(sc2::UNIT_TYPEID::INVALID), time(time) {}

    /** True if this event may have an impact on food or mining speed */
    bool impactsEconomy() const;

    /** Applies the effects of this event on the given state */
    void apply(BuildState& state) const;

    inline bool operator<(const BuildEvent& other) const {
        return time < other.time;
    }
};

struct BuildOrderFitness {
    static const BuildOrderFitness ReallyBad;

    float time;
    BuildResources resources;
    MiningSpeed miningSpeed;

    BuildOrderFitness () : time(0), resources(0,0), miningSpeed({0,0}) {}
    BuildOrderFitness (float time, BuildResources resources, MiningSpeed miningSpeed) : time(time), resources(resources), miningSpeed(miningSpeed) {}

    float score() const;

    bool operator<(const BuildOrderFitness& other) const {
        return score() < other.score();
    }
};

struct BaseInfo {
    float remainingMinerals;
    float remainingVespene1;
    float remainingVespene2;

    BaseInfo (float minerals, float vespene1, float vespene2) : remainingMinerals(minerals), remainingVespene1(vespene1), remainingVespene2(vespene2) {}

    inline void mineMinerals(float amount) {
        remainingMinerals = fmax(0.0f, remainingMinerals - amount);
    }

    /** Returns (high yield, low yield) mineral slots on this expansion */
    inline std::pair<int,int> mineralSlots () const {
        // Max is 10800 for an expansion with 8 patches
        if (remainingMinerals > 4800) return {16, 8};
        if (remainingMinerals > 4000) return {12, 6};
        if (remainingMinerals > 100) return {8, 4};
        if (remainingMinerals > 0) return {2, 1};
        return {0, 0};
    }
};

struct BuildOrderState {
    const std::vector<sc2::UNIT_TYPEID>& buildOrder;
    int buildIndex = 0;

    BuildOrderState (const std::vector<sc2::UNIT_TYPEID>& buildOrder) : buildOrder(buildOrder) {}
};

/** Represents all units, buildings and current build/train actions that are in progress for a given player */
struct BuildState {
    /** Time in game time seconds at normal speed */
    float time = 0;
    /** Race of the player */
    sc2::Race race = sc2::Race::Terran;

    /** All units in the current state */
    std::vector<BuildUnitInfo> units;
    /** All future events, sorted in ascending order by their time */
    std::vector<BuildEvent> events;
    /** Current resources */
    BuildResources resources = BuildResources(0,0);
    /** Metadata (in particular resource info) about the bases that the player has */
    std::vector<BaseInfo> baseInfos;

    BuildState() {}
    BuildState(std::vector<std::pair<sc2::UNIT_TYPEID, int>> unitCounts) {
        for (auto u : unitCounts)
            addUnits(u.first, u.second);
    }
    BuildState(const sc2::ObservationInterface* observation, sc2::Unit::Alliance alliance, sc2::Race race, BuildResources resources, float time);

    /** Marks a number of units with the given type (and optionally addon) as being busy.
     * Delta may be negative to indicate that the units should be made available again after having been busy.
     * 
     * If this action could not be performed (e.g. there were no non-busy units that could be made busy) then the function will panic.
     */
    void makeUnitsBusy(sc2::UNIT_TYPEID type, sc2::UNIT_TYPEID addon, int delta);

    void addUnits(sc2::UNIT_TYPEID type, int delta);

    /** Adds a number of given unit type (with optional addon) to the state.
     * Delta may be negative to indicate that units should be removed.
     * 
     * If this action could not be performed (e.g. there were no units that could be removed) then this function will panic.
     */
    void addUnits(sc2::UNIT_TYPEID type, sc2::UNIT_TYPEID addon, int delta);

    /** Returns the current mining speed of (minerals,vespene gas) per second (at normal game speed) */
    MiningSpeed miningSpeed() const;

    /** Returns the time it will take to get the specified resources using the given mining speed */
    float timeToGetResources(MiningSpeed miningSpeed, float mineralCost, float vespeneCost) const;

    /** Adds a new future event to the state */
    void addEvent(BuildEvent event);

    /** Simulate the state until a given point in time.
     * All actions up to and including the end time will have been completed after the function has been called.
     * This will update the current resources using the simulated mining speed.
     */
    void simulate(float endTime);

    /** Simulates the execution of a given build order.
     * The function will return false if the build order could not be performed.
     * The resulting state is the state right after the final item in the build order has been completed.
     * 
     * The optional callback will be called once exactly when build order item number N is executed, with N as the parameter.
     * Note that the this is when the item starts to be executed, not when the item is finished.
     * The callback is called right after the action has been executed, but not necessarily completed.
     */
    bool simulateBuildOrder(const std::vector<sc2::UNIT_TYPEID>& buildOrder, std::function<void(int)> = nullptr, bool waitUntilItemsFinished = true);
    bool simulateBuildOrder(BuildOrderState& buildOrder, std::function<void(int)> callback, bool waitUntilItemsFinished, float maxTime = std::numeric_limits<float>::infinity());

    /** Food that is currently available.
     * Positive if there is a surplus of food.
     * Note that food is a floating point number, zerglings in particular use 0.5 food.
     * It is still safe to work with floating point numbers because they can exactly represent whole numbers and whole numbers + 0.5 exactly up to very large values.
     */
    float foodAvailable() const;

    /** True if the state contains the given unit type or which is equivalent to the given unit type for tech purposes */
    bool hasEquivalentTech(sc2::UNIT_TYPEID type) const;
};

std::pair<std::vector<sc2::UNIT_TYPEID>, std::vector<bool>> expandBuildOrderWithImplicitSteps (const BuildState& startState, std::vector<sc2::UNIT_TYPEID> buildOrder);

std::vector<sc2::UNIT_TYPEID> findBestBuildOrderGenetic(const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& startingUnits, const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& target);
std::vector<sc2::UNIT_TYPEID> findBestBuildOrderGenetic(const BuildState& startState, const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& target, const std::vector<sc2::UNIT_TYPEID>* seed = nullptr);
void unitTestBuildOptimizer();
void printBuildOrderDetailed(const BuildState& startState, std::vector<sc2::UNIT_TYPEID> buildOrder, const std::vector<bool>* highlight = nullptr);