#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"
#include "Group.h"

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

class GroupActionNode : public BOT::ContextAwareActionNode {
public:
    GroupActionNode(BOT::Context* context) : ContextAwareActionNode(context) {}
};

class GroupConditionNode : public BOT::ContextAwareConditionNode {
public:
    GroupConditionNode(BOT::Context* context) : BOT::ContextAwareConditionNode(context){}
};

class GroupAttackMove : public GroupActionNode {
public:
    GroupAttackMove(BOT::Context* group) : GroupActionNode(group) {}
    BOT::Status OnTick() override;
};

class ScoutingBehavior: public GroupActionNode{
public:
    ScoutingBehavior(BOT::Context* group) : GroupActionNode(group) {}
    BOT::Status OnTick() override;

};
