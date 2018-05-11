#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"

class ControlSupplyDepots : public BOT::ActionNode {
public:
    BOT::Status OnTick() override;
};

class SimpleAttackMove : public BOT::ActionNode {
    BOT::Status OnTick() override;
};