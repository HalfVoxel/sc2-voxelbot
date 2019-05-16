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
    ArmyConsolidate,
    ArmyMoveC1,
    ArmyMoveC2,
    ArmyMoveC3,
    None,
    ArmyMoveBase,
    ArmyAttackBase,
    Reinforce,

    IdleArmyAttackClosestEnemy,
    IdleArmyConsolidate,
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
            case MCTSAction::ArmyMoveC3:
            return "ArmyMoveC3";
        case MCTSAction::None:
            return "None";
        case MCTSAction::ArmyMoveBase:
            return "ArmyMoveBase";
        case MCTSAction::ArmyAttackBase:
            return "ArmyAttackBase";
            case MCTSAction::Reinforce:
            return "Reinforce";
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
    std::array<MCTSAction, 2> lastActions = {{ MCTSAction::None, MCTSAction::None }};

    SimulatorMCTSState (SimulatorState state, int player = 0) : state(state), player(player) {}

    std::pair<SimulatorMCTSState, bool> step(int action);

    int isWin() const;
    bool executeAction(MCTSAction action, std::function<void(SimulatorUnitGroup&, SimulatorOrder)>* commandListener = nullptr);
    bool internalStep(int action, bool ignoreUnintentionalNOOP=false);

    std::vector<std::pair<int, float>> generateMoves();

    std::array<float, 2> rollout() const;
    std::string to_string() const;
};

struct MCTSSearchSC2 {
    std::unique_ptr<MCTSSearch<int, SimulatorMCTSState>> search = nullptr;
    std::shared_ptr<SimulatorContext> simulator = nullptr;

    MCTSSearchSC2() {}
    MCTSSearchSC2(std::unique_ptr<MCTSSearch<int, SimulatorMCTSState>>& search, std::shared_ptr<SimulatorContext>& simulator) : search(move(search)), simulator(simulator) {}
};

MCTSSearchSC2 findBestActions(SimulatorState& startState, int startingPlayerIndex);
