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

    // TODO: Reinforce
    
    // Not an action, just a size indicator
    Count
};

inline std::string MCTSActionName(MCTSAction action) {
    switch(action) {
        case MCTSAction::ArmyAttackClosestEnemy:
            return "ArmyAttackClosestEnemy";
        case MCTSAction::IdleArmyAttackClosestEnemy:
            return "IdleArmyAttackClosestEnemy";
        case MCTSAction::ArmyConsolidate:
            return "ArmyConsolidate";
        case MCTSAction::IdleArmyConsolidate:
            return "IdleArmyConsolidate";
        case MCTSAction::ArmyMoveC1:
            return "ArmyMoveC1";
        case MCTSAction::ArmyMoveC2:
            return "ArmyMoveC2";
        case MCTSAction::None:
            return "None";
        case MCTSAction::ArmyMoveBase:
            return "ArmyMoveBase";
        case MCTSAction::ArmyAttackBase:
            return "ArmyAttackBase";
        case MCTSAction::ArmySuicide:
            return "ArmySuicide";
        case MCTSAction::IdleNonArmyAttackClosestEnemy:
            return "IdleNonArmyAttackClosestEnemy";
        case MCTSAction::NonArmyAttackClosestEnemy:
            return "NonArmyAttackClosestEnemy";
        case MCTSAction::NonArmyMoveBase:
            return "NonArmyMoveBase";
        default:
            return "Invalid";
    }
}

extern std::function<bool(const SimulatorUnitGroup&)> idleGroup;
extern std::function<bool(const SimulatorUnitGroup&)> structureGroup;
extern std::function<bool(const SimulatorUnit&)> armyUnit;
extern std::function<bool(const SimulatorUnit&)> notArmyUnit;
extern std::function<bool(const SimulatorUnit&)> structureUnit;

float healthFraction(const SimulatorState& state, int owner);
sc2::Point2D averagePos(std::vector<SimulatorUnitGroup*> groups);

struct SimulatorMCTSState {
    SimulatorState state;
    int player = 0;
    int count = 0;
    int seed = rand();

    SimulatorMCTSState (SimulatorState state, int player = 0) : state(state), player(player) {}

    std::pair<SimulatorMCTSState, bool> step(int action);

    int isWin() const;
    bool executeAction(MCTSAction action, std::function<void(SimulatorUnitGroup&, SimulatorOrder)>* commandListener = nullptr);
    bool internalStep(int action, bool ignoreUnintentionalNOOP=false);

    std::vector<std::pair<int, float>> generateMoves();

    float rollout() const;
    std::string to_string() const;
};

std::unique_ptr<MCTSState<int, SimulatorMCTSState>> findBestActions(SimulatorState& startState, int startingPlayerIndex);
