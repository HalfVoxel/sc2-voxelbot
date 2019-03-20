#pragma once
#include "simulator.h"
#include "../mcts/mcts.h"
#include<vector>
#include<functional>
#include<string>

struct SimulatorState;
struct SimulatorUnit;

enum class MCTSAction {
    ArmyAttackClosestEnemy,
    IdleArmyAttackClosestEnemy,
    ArmyConsolidate,
    IdleArmyConsolidate,
    ArmyMoveC1,
    ArmyMoveC2,
    None,
    ArmyMoveBase,
    ArmyAttackBase,
    ArmySuicide,
    IdleNonArmyAttackClosestEnemy,
    NonArmyAttackClosestEnemy,
    NonArmyMoveBase,
};

extern std::function<bool(const SimulatorUnitGroup&)> idleGroup;
extern std::function<bool(const SimulatorUnitGroup&)> structureGroup;
extern std::function<bool(const SimulatorUnit&)> armyUnit;
extern std::function<bool(const SimulatorUnit&)> notArmyUnit;
extern std::function<bool(const SimulatorUnit&)> structureUnit;

float healthFraction(const SimulatorState& state, int owner);

struct SimulatorMCTSState {
    int player = 0;
    SimulatorState state;
    int count = 0;

    SimulatorMCTSState (SimulatorState state) : state(state) {}

    std::pair<SimulatorMCTSState, bool> step(int action);

    int isWin() const;
    bool executeAction(MCTSAction action, std::function<void(SimulatorUnitGroup&, SimulatorOrder)>* commandListener = nullptr);
    bool internalStep(int action, bool ignoreUnintentionalNOOP=false);

    std::vector<std::pair<int, float>> generateMoves();

    float rollout() const;
    std::string to_string() const;
};

std::unique_ptr<State<int, SimulatorMCTSState>> findBestActions(SimulatorState& startState);
