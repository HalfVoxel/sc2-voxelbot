#include <vector>
#include "sc2api/sc2_interfaces.h"

struct CombatUnit;
struct SimulatorState;

enum class GeneSelector {
    No,
    Yes,
    Irrelevant
};

enum class GenePolicyAction {
    Attack,
    Move,
};

enum class GenePolicyMovementTarget {
    ClosestEnemy,
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

    ContextuallyRelevantPoint,
};

enum class GeneConditionType {
    EnemiesNearby,
    EnemiesAttacking,
    ExpectedWinAgainstEnemies,
    ExpectedLossAgainstEnemies,
};

struct GeneCondition {
    int group;
    GenePolicyMovementTarget point;
    std::vector<int> activatedPolicies;

    // Threshold of when this condition triggers
    // For enemies this is measured in number of units
    float threshold = 0;

    bool isTriggered(const SimulatorState& state);
};

struct GeneGroupDefinition {
    GeneSelector army = GeneSelector::Irrelevant;
    GeneSelector flying = GeneSelector::Irrelevant;
    // Invalid for all
    sc2::UNIT_TYPEID unitType = sc2::UNIT_TYPEID::INVALID;
    // 0 for all
    int count = 0;

    bool isMatching(const CombatUnit& unit);
    float entropy() {
        float e = 0;
        if (army != GeneSelector::Irrelevant) e += 1; // Log2(2)
        if (flying != GeneSelector::Irrelevant) e += 1;
        if (unitType != sc2::UNIT_TYPEID::INVALID) e += 5; // ≈log2(30), sort of how many unit/building types there are
        if (count != 0) e += 1;

        return e;
    }
};

struct GenePolicyStep {
    GenePolicyAction action;
    GenePolicyMovementTarget target;
};

struct GenePolicy {
    int group;
    std::vector<GenePolicyStep> steps;
    bool repeatable;
};

struct GenePlan {
    std::vector<GeneGroupDefinition> groups;
    std::vector<GenePolicy> policies;
    std::vector<GeneCondition> conditions;
};

struct GeneticPlanner {
    GeneticPlanner(const SimulatorState& startingState);

    GenePlan plan();
    void mutate(GenePlan& plan);
    void evaluate(SimulatorState state, const GenePlan& player1plan, const GenePlan& player2plan);
};
