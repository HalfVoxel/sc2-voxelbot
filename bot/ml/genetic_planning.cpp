#if FALSE
#include "genetic_planning.h"
#include "simulator.h"
#include "simulator_context.h"
#include "../CombatPredictor.h"
#include "../utilities/stdutils.h"


using namespace std;
using namespace sc2;


GenePlan GeneticPlanner::plan(const SimulatorState& startingState) {

    vector<GenePlan> generation;
    const int iterations = 100;

    for (int it = 0; it < iterations; it++) {

        calculateFitness(generation[i], enemyStrategies[random]);
    }
}

Point2D calculateMovementTarget(const SimulatorState& state, vector<SimulatorUnitGroup*> groups, GenePolicyMovementTarget target) {
    assert(groups.size() > 0);

    int playerID = groups[0].owner;
    int opponentID = 3 - playerID;
    switch(target) {
        case GenePolicyMovementTarget::ClosestEnemy: {
            auto avgPos = averagePos(groups);
            auto* closestEnemy = closestGroup(state, opponentID, avgPos, nullptr);
            if (closestEnemy != nullptr) return closestEnemy->pos;
            break;
        }
        case GenePolicyMovementTarget::ClosestEnemyBase: {
            auto avgPos = averagePos(groups);
            auto* closestEnemy = closestGroup(state, opponentID, avgPos, structureGroup);
            if (closestEnemy != nullptr) return closestEnemy->pos;
            break;
        }
        case GenePolicyMovementTarget::ClosestAllyBase: {
            auto avgPos = averagePos(groups);
            auto* closestEnemy = closestGroup(state, playerID, avgPos, structureGroup);
            if (closestEnemy != nullptr) return closestEnemy->pos;
            break;
        }
        case GenePolicyMovementTarget::GroupAverage:
            return averagePos(groups);
    }

    return Point2D(0, 0);
}

/*ClosestEnemy,
ClosestEnemyBase,
ClosestAllyBase,
GroupAverage,
EnemyBase1,
EnemyBase2,
EnemyBase3,
EnemyBase4,
EnemyBase5,
P1,
P2,
P3,
P4,
P5,
OurBase1,
OurBase2,
OurBase3,
OurBase4,
OurBase5,

// Note: randomized in case of an action, otherwise mostly used as a condition
OurBaseAny,

// Quadrants of the map
Quad1,
Quad2,
Quad3,
Quad4,
*/

struct GenePlanExecutor {
    const GenePlan& plan;
    int player;

    GenePlanExecutor(const GenePlan& plan, int player) : plan(plan), player(player) {}

    void executePolicy(SimulatorState& state, const GenePolicy& policy) {
        // Determine group
        // Execute step
        auto& group = plan.groups[policy.group];
        vector<SimulatorUnitGroup*> selection = state.select(player, nullptr, [&](const SimulatorUnit& unit) { return group.isMatching(unit) });
        calculateMovementTarget(state, selection, policy.target);
    }

    void execute(SimulatorState& state) {
        // Find out which policies are active
        // Sort policies
        // Execute policies in order
        vector<int> policyScores(state.policies.size());
        vector<int> policySortOrder(state.policies.size());
        for (int i = 0; i < state.policies.size(); i++) {
            policyScores[i] = state.groups[state.policies[i].group].entropy() * 0.001f;
            policySortOrder[i] = i;
        }

        for (int i = 0; i < state.conditions.size(); i++) {
            if (state.conditions[i].isTriggered(state)) {
                for (auto p : state.conditions[i].activatedPolicies) {
                    policyScores[p]++;
                }
            }
        }

        sortByValueDescending(policySortOrder, [&](int index) { return policyScores[index] });

        for (int i = 0; i < state.policies.size(); i++) {
            auto& policy = state.policies[policySortOrder[i]];
            executePolicy(state, policy);
        }
    }
};

GeneticPlanner::evaluate(SimulatorState state, const GenePlane& player1plan, const GenePlan& player2plan) {
    GenePlanExecutor player1(player1plan, 1);
    GenePlanExecutor player2(player2plan, 2);
    float maxTime = state.time() + 60 * 20;

    while(true) {
        player1.execute(state);
        player2.execute(state);
        state.simulate(state.time() + 5);

        if (state.time() > maxTime) break;

        float healthFraction = healthFraction(state, 1);
        if (healthFraction >= 0.99f || healthFraction <= 0.01f) break;
    }

    return healthFraction(state, 1) > 0.5f;
}

GenePlanFitness calculateFitness(const GenePlan& strategy, const vector<GenePlan&> enemyStrategies, const vector<SimulatorState&>& startingStates) {
    assert(enemyStrategies.size() == startingStates.size());

    for (int i = 0; i < enemyStrategies.size(); i++) {
        auto result = evaluate(startingStates[i], strategy, enemyStrategies[i]);

    }
}

#endif