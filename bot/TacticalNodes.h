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

class TacticalMove : public GroupActionNode {
    int pathingTicker = 0;
    std::vector<sc2::Point2DI> currentPath;

public:
    TacticalMove(BOT::Context* group) : GroupActionNode(group) {}
    BOT::Status OnTick() override;
};

class GroupPosition : public GroupActionNode {
public:
    GroupPosition(BOT::Context* group) : GroupActionNode(group) {}
    BOT::Status OnTick() override;
};


class InCombat: public GroupConditionNode{
public:
    InCombat(BOT::Context* group) : GroupConditionNode(group) {}
    BOT::Status OnTick() override;
};

class GroupAttackMove : public GroupActionNode {
public:
    GroupAttackMove(BOT::Context* group) : GroupActionNode(group) {}
    BOT::Status OnTick() override;
};

class StrikeGroupBehavior : public BOT::SelectorNode {
public:
    StrikeGroupBehavior(BOT::Context* group)
        : BOT::SelectorNode({
            new BOT::SequenceNode{
                new InCombat(group),
                new GroupAttackMove(group)
            },
            new TacticalMove(group)
        }) {
    }
};

class MainGroupBehavior : public BOT::SelectorNode {
public:
    MainGroupBehavior(BOT::Context* group)
        : BOT::SelectorNode({
            new BOT::SequenceNode{
                new InCombat(group),
                new GroupAttackMove(group)
            },
            new GroupPosition(group)
        }) {
    }
};
