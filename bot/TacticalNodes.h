#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"

class ControlSupplyDepots : public BOT::ActionNode {
    BOT::Status OnTick() override;
};

class SimpleArmyPosition : public BOT::ActionNode {
    BOT::Status OnTick() override;
};

class SimpleAttackMove : public BOT::ActionNode {
    BOT::Status OnTick() override;
};

class IsUnderAttack: public BOT::ConditionNode {
    BOT::Status OnTick() override;
};