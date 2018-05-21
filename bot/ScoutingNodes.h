#pragma once
#include "TacticalNodes.h"

class ScoutingBehavior : public GroupActionNode {
public:
	int timer = 0;
    ScoutingBehavior(BOT::Context* group) : GroupActionNode(group) {}
    BOT::Status OnTick() override;
};
