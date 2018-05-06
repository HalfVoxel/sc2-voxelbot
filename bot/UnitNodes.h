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
	sc2::Tag location;

protected:
    sc2::UnitTypeID builderUnitType;
    sc2::AbilityID abilityType;

public:
    BOT::Status PlaceBuilding(sc2::ABILITY_ID ability, sc2::UNIT_TYPEID unitType, sc2::Tag loc);
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

class BuildGas : public BuildStructure {

public:
	BuildGas(sc2::AbilityID ability, sc2::UnitTypeID unit) : BuildStructure(ability, unit) {}
	BOT::Status OnTick() override;
};


class AssignHarvesters : public BOT::ActionNode {
    sc2::UnitTypeID workerUnitType;
    sc2::AbilityID abilityType;
    sc2::UNIT_TYPEID gasBuildingType;
public:
    BOT::Status OnTick() override;
    BOT::Status MineIdleWorkers(const sc2::Unit* worker, sc2::AbilityID worker_gather_command, sc2::UnitTypeID vespene_building_type);
    BOT::Status ManageWorkers(sc2::UNIT_TYPEID worker_type, sc2::AbilityID worker_gather_command, sc2::UNIT_TYPEID vespene_building_type);
    AssignHarvesters(sc2::UnitTypeID workerUnitType,   sc2::AbilityID abilityType,  sc2::UNIT_TYPEID gasBuildingType) : workerUnitType(workerUnitType), abilityType(abilityType), gasBuildingType(gasBuildingType) {}
    const sc2::Unit* FindNearestMineralPatch(const sc2::Point2D& start);
};