#pragma once
#include "../CombatPredictor.h"
#include "simulator.h"
#include "mcts_sc2.h"
#include "../utilities/bump_allocator.h"
#include<numeric>


struct CombatHasher {
    uint64_t hash = 0;

    inline void hashUnit(const CombatUnit& unit) {
        hash = (hash * 31) ^ (unsigned long long)unit.energy;
        hash = (hash * 31) ^ (unsigned long long)unit.health;
        hash = (hash * 31) ^ (unsigned long long)unit.shield;
        hash = (hash * 31) ^ (unsigned long long)unit.type;
        hash = (hash * 31) ^ (unsigned long long)unit.owner;
    }

    inline void hashParams(int defenderPlayer, bool badMicro, float maxTime) {
        hash = hash ^ defenderPlayer;
        hash = hash*31 ^ (int)badMicro;
        hash = hash*31 ^ (int)round(maxTime);
    }
};

struct SimulationCacheItem {
    BuildState* buildState;
    BuildOrderState buildOrder;
    std::vector<BuildEvent> buildEvents;

    SimulationCacheItem (BuildState* buildState, BuildOrderState buildOrder, std::vector<BuildEvent> buildEvents) : buildState(buildState), buildOrder(buildOrder), buildEvents(buildEvents) {}
};

struct MCTSCache {
    BumpAllocator<BuildState> buildStateAllocator;
    std::map<uint64_t, CombatResult> combatResults;
    std::map<uint64_t, SimulationCacheItem> simulationCache;
    std::map<uint64_t, std::pair<const BuildState*, const BuildState*>> stateCombatTransitions;

    BuildState* copyState(const BuildState& state);

    void applyCombatOutcome(SimulatorState& state, const std::vector<SimulatorUnitGroup*>& groups, const CombatResult& outcome);

    /** Simulate a build state with a given build order, but return an existing cached state if possible */
    std::pair<const BuildState*, BuildOrderState> simulateBuildOrder(const BuildState& state, const BuildOrderState& buildOrder, float endTime, const std::function<void(const BuildEvent&)>* listener);
    void handleCombat(SimulatorState& state, const std::vector<SimulatorUnitGroup*>& groups, int defender, float maxTime, bool debug);
    void clear();
};
