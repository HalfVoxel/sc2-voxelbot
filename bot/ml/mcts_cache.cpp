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

BuildState* MCTSCache::copyState(const BuildState& state) {
    // auto newState = make_shared<BuildState>(state);
    return buildStateAllocator.allocate(state);
}

void applyReward(SimulatorState& state, const CombatUnit& oldUnit, const CombatUnit& newUnit) {
    float healthDelta = (newUnit.health + newUnit.shield) - (oldUnit.health + oldUnit.shield);

    if (healthDelta != 0) {
        // Temporary unit, don't give rewards for it
        if (newUnit.type == UNIT_TYPEID::ZERG_BROODLING) return;

        // Additional penalty for actually killing a unit
        if (newUnit.health == 0) healthDelta -= (maxHealth(newUnit.type) + maxShield(newUnit.type)) * 0.25f;

        float rewardDecay = exp(-(state.time() - shared_ptr<SimulatorContext>(state.simulator)->simulationStartTime) / 200.0f);
        // Kill rewards decay, losses do not
        // Loosing structures is really bad
        int playerIndex = newUnit.owner - 1;
        int opponentIndex = 1 - playerIndex;

        if (shared_ptr<SimulatorContext>(state.simulator)->debug) {
            cout << "Reward " << healthDelta << " " << rewardDecay << endl;
        }

        state.rewards[playerIndex] += healthDelta * (isStructure(newUnit.type) ? 3.0f : 1.0f);
        state.rewards[opponentIndex] -= healthDelta * rewardDecay;
    }
}

void MCTSCache::applyCombatOutcome(SimulatorState& state, const vector<SimulatorUnitGroup*>& groups, const CombatResult& outcome) {
    array<BuildState*, 2> newStates = {{ copyState(*state.states[0]), copyState(*state.states[1]) }};
    state.states[0] = newStates[0];
    state.states[1] = newStates[1];

    int index = 0;
    for (auto* group : groups) {
        for (auto& u : group->units) {
            auto& newUnit = outcome.state.units[index];
            applyReward(state, u.combat, newUnit);
            u.combat = newUnit;
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

pair<const BuildState*, BuildOrderState> MCTSCache::simulateBuildOrder(const BuildState& state, const BuildOrderState& buildOrder, float endTime, const std::function<void(const BuildEvent&)>* listener) {
    uint64_t hash = state.immutableHash();
    hash = hash*31 ^ ((int)(endTime*1000));
    hash = hash*31 ^ buildOrder.buildIndex;
    hash = hash*31 ^ (uint64_t)(&*buildOrder.buildOrder);
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

void MCTSCache::handleCombat(SimulatorState& state, const vector<SimulatorUnitGroup*>& groups, int defender, float startTime, float maxTime, bool debug) {
    bool badMicro = false;

    CombatHasher combatHasher;
    combatHasher.hashParams(defender, badMicro, startTime, maxTime);
    for (auto* group : groups) {
        for (auto& u : group->units) combatHasher.hashUnit(u.combat);
    }

    uint64_t stateCombatTransitionHash = (combatHasher.hash * 31 ^ state.states[0]->immutableHash()) * 31 ^ state.states[1]->immutableHash();

    auto combatIt = combatResults.find(combatHasher.hash);
    combatTot++;
    if (combatIt != combatResults.end() && !debug) {
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
                    auto& newUnit = outcome.state.units[index];
                    assert(u.combat.type == newUnit.type);
                    applyReward(state, u.combat, newUnit);
                    u.combat = newUnit;
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
        settings.debug = debug;
        settings.workersDoNoDamage = true;
        settings.startTime = startTime;
        CombatResult combatResult = shared_ptr<SimulatorContext>(state.simulator)->combatPredictor->predict_engage(combatState, settings, nullptr, defender);
        applyCombatOutcome(state, groups, combatResult);

        combatResults[combatHasher.hash] = combatResult;
        stateCombatTransitions[stateCombatTransitionHash] = make_pair(state.states[0], state.states[1]);
    }
}
