#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"
#include "Group.h"
#include "StrategicNodes.h"

class GroupActionNode : public BOT::ContextAwareActionNode {
protected:
    UnitGroup * GetGroup() { return (UnitGroup*)context; }
public:
    GroupActionNode(BOT::Context* context) : ContextAwareActionNode(context) {}
};

class GroupConditionNode : public BOT::ContextAwareConditionNode {
protected:
    UnitGroup * GetGroup() { return (UnitGroup*)context; }
public:
    GroupConditionNode(BOT::Context* context) : BOT::ContextAwareConditionNode(context) {}
};

class ControlSupplyDepots : public BOT::ActionNode {
    BOT::Status OnTick() override;
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

class GroupPosition : public GroupActionNode {
public:
    GroupPosition(BOT::Context* group) : GroupActionNode(group) {}
    BOT::Status OnTick() override;
};

class MainGroupBehavior : public BOT::ParallelNode {
public:
    MainGroupBehavior(BOT::Context* group): ParallelNode({
            new GroupPosition(group),
            new BOT::SequenceNode{
                new HasUnit(sc2::UNIT_TYPEID::TERRAN_MARINE, 40),
                new GroupAttackMove(group)
            }
        }) {
    }
};
