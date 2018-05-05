#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"

class BuildUnit : public BOT::TreeNode {
	sc2::UnitTypeID unitType;
	sc2::AbilityID abilityType;
public:
	BuildUnit(sc2::AbilityID ability, sc2::UnitTypeID unit) : abilityType(ability), unitType(unit) {}
	BOT::Status Tick();
};