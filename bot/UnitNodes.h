#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"

class BuildUnit : public BOT::ActionNode {
	sc2::UnitTypeID unitType;
	sc2::AbilityID abilityType;
public:
	BuildUnit(sc2::AbilityID ability, sc2::UnitTypeID unit) : abilityType(ability), unitType(unit) {}
	BOT::Status OnTick() override;
};

class BuildStructure : public BOT::ActionNode {
	sc2::UnitTypeID builderUnitType;
	sc2::AbilityID abilityType;
	sc2::Tag location;
public:
	BuildStructure(sc2::AbilityID ability, sc2::UnitTypeID unit) : abilityType(ability), builderUnitType(unit), location(sc2::NullTag) {}
	BuildStructure(sc2::AbilityID ability, sc2::UnitTypeID unit, sc2::Tag location) : abilityType(ability), builderUnitType(unit), location(location) {}
	BOT::Status OnTick() override;
};

class HasUnit : public BOT::ConditionNode {
	sc2::UnitTypeID unit;
	int count;
public:
	HasUnit(sc2::UnitTypeID unit, int count=1) : unit(unit), count(count) {}
	BOT::Status OnTick() override;
};

class ShouldBuildSupply : public BOT::ConditionNode {
public:
	ShouldBuildSupply() {}
	BOT::Status OnTick() override;
};

class BuildGas : public BOT::ActionNode {
	std::unique_ptr<BOT::TreeNode> child;

public:
	BuildGas() {}
	BOT::Status OnTick() override;
};
