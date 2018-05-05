#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"

class BuildUnit : public BOT::TreeNode {
	sc2::UnitTypeID unit;
public:
	BuildUnit(sc2::UnitTypeID unit) : unit(unit) {}
	BOT::Status Tick();
};