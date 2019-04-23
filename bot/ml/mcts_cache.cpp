#include "mcts_cache.h"
#include "sc2lib/sc2_lib.h"
#include "simulator_context.h"

using namespace std;
using namespace sc2;

void MCTSCache::clear() {
    combatResults.clear();
    simulationCache.clear();
    stateCombatTransitions.clear();
}

shared_ptr<BuildState> MCTSCache::copyState(const BuildState& state) {
    auto newState = make_shared<BuildState>(state);
    return newState;
}

void MCTSCache::applyCombatOutcome(SimulatorState& state, const vector<SimulatorUnitGroup*>& groups, const CombatResult& outcome) {
    array<std::shared_ptr<BuildState>, 2> newStates = {{ copyState(*state.states[0]), copyState(*state.states[1]) }};
    state.states[0] = newStates[0];
    state.states[1] = newStates[1];

    int index = 0;
    for (auto* group : groups) {
        for (auto& u : group->units) {
            u.combat = outcome.state.units[index];
            index++;

            if (u.combat.health <= 0) {
                // TODO: Handle addons?
                newStates[u.combat.owner - 1]->killUnits(u.combat.type, UNIT_TYPEID::INVALID, 1);
            }
        }

        state.filterDeadUnits(group);
    }

    newStates[0]->recalculateHash();
    newStates[1]->recalculateHash();
    assert(state.states[0]->immutableHash() == state.states[0]->hash());
    state.assertValidState();
}

int simHits;
int simTot;

int combatHits;
int combatTot;

const bool debugMCTSCache = false;

pair<shared_ptr<const BuildState>, BuildOrderState> MCTSCache::simulateBuildOrder(const BuildState& state, const BuildOrderState& buildOrder, float endTime, const std::function<void(const BuildEvent&)>* listener) {
    uint64_t hash = state.immutableHash();
    hash = hash*31 ^ ((int)(endTime*1000));
    hash = hash*31 ^ buildOrder.buildIndex;
    hash = hash*31 ^ (uint64_t)(&buildOrder.buildOrder);
    // cout << "Simulating " << state.immutableHash() << " to " << endTime << endl;
    simTot++;
    if (simTot % 5000 == 0) {
        // cout << "Stats " << simHits << "/" << simTot << " " << combatHits << "/" << combatTot << endl;
    }

    auto ptr = simulationCache.find(hash);
    if (ptr == simulationCache.end()) {
        // cout << "No simulation cache hit" << endl;
        // if (ptr != simulationCache.end()) simulationCache.erase(ptr);

        SimulationCacheItem& newCache = simulationCache.insert(make_pair(hash, SimulationCacheItem(copyState(state), buildOrder, {}))).first->second;
        std::function<void(const BuildEvent&)> tmpListener = [&](const BuildEvent& ev) {
            newCache.buildEvents.push_back(ev);
        };

        newCache.buildState->simulateBuildOrder(newCache.buildOrder, nullptr, false, endTime, &tmpListener);

        // If the build order is finished, simulate until the end time anyway
        if (newCache.buildState->time < endTime) newCache.buildState->simulate(endTime, &tmpListener);

        newCache.buildState->recalculateHash();
        ptr = simulationCache.find(hash);
        assert(ptr != simulationCache.end());
    } else {
        simHits++;
        // cout << "Simulation cache hit" << endl;
    }

    SimulationCacheItem& cache = ptr->second;

    for (auto& e : cache.buildEvents) (*listener)(e);

    if (debugMCTSCache) assert(cache.buildState->immutableHash() == cache.buildState->hash());
    return make_pair(cache.buildState, cache.buildOrder);
}

void MCTSCache::handleCombat(SimulatorState& state, const vector<SimulatorUnitGroup*>& groups, int defender, float maxTime) {
    bool badMicro = false;

    CombatHasher combatHasher;
    combatHasher.hashParams(defender, badMicro, maxTime);
    for (auto* group : groups) {
        for (auto& u : group->units) combatHasher.hashUnit(u.combat);
    }

    uint64_t stateCombatTransitionHash = (combatHasher.hash * 31 ^ state.states[0]->immutableHash()) * 31 ^ state.states[1]->immutableHash();

    auto combatIt = combatResults.find(combatHasher.hash);
    combatTot++;
    if (combatIt != combatResults.end()) {
        combatHits++;
        // cout << "Combat result cached" << endl;
        const CombatResult& outcome = combatIt->second;

        auto transitionIt = stateCombatTransitions.find(stateCombatTransitionHash);
        
        if (transitionIt != stateCombatTransitions.end()) {
            // cout << "Combat transition cached" << endl;
            // Both the combat outcome and the new simulation states are cached
            state.states[0] = transitionIt->second.first;
            state.states[1] = transitionIt->second.second;

            // Apply the combat result without modifying the simulation states as they are
            // assumed to be up to date already
            int index = 0;
            for (auto* group : groups) {
                for (auto& u : group->units) {
                    assert(u.combat.type == outcome.state.units[index].type);
                    u.combat = outcome.state.units[index];
                    index++;
                }

                state.filterDeadUnits(group);
            }

            if (debugMCTSCache) assert(state.states[0]->immutableHash() == state.states[0]->hash());

            state.assertValidState();
        } else {
            // cout << "Combat transition NOT cached" << endl;
            // Only the combat outcome is cached
            applyCombatOutcome(state, groups, outcome);
            stateCombatTransitions[stateCombatTransitionHash] = make_pair(state.states[0], state.states[1]);
            if (debugMCTSCache) assert(state.states[0]->immutableHash() == state.states[0]->hash());
        }
    } else {
        // cout << "No cache hit" << endl;
        // No cache hits at all.
        // Calculate the combat result from scratch

        CombatState combatState;
        for (auto* group : groups) {
            for (auto& u : group->units) combatState.units.push_back(u.combat);
        }

        // TODO: If both armies had a fight the previous time step as well, then they should already be in position (probably)
        // TODO: What if the combat drags on longer than to endTime? (probably common in case of harassment, it takes some time for the player to start to defend)
        // Add a max time to predict_engage and stop. Combat will resume next simulation step.
        // Note: have to ensure that combat is resumed properly (without the attacker having to move into range and all that)
        CombatSettings settings;
        settings.maxTime = maxTime;
        settings.badMicro = badMicro;
        CombatResult combatResult = shared_ptr<SimulatorContext>(state.simulator)->combatPredictor->predict_engage(combatState, settings, nullptr, defender);
        applyCombatOutcome(state, groups, combatResult);

        combatResults[combatHasher.hash] = combatResult;
        stateCombatTransitions[stateCombatTransitionHash] = make_pair(state.states[0], state.states[1]);
    }
}
