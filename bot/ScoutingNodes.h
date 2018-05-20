#pragma once
#include "TacticalNodes.h"

class ScoutingBehavior : public GroupActionNode {
public:
    ScoutingBehavior(BOT::Context* group) : GroupActionNode(group) {}
    BOT::Status OnTick() override;
};
