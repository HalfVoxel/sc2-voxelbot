#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"

class BuildUnit : public BOT::ActionNode {
	sc2::UnitTypeID unitType;
public:
	BuildUnit(sc2::UnitTypeID unit) : unitType(unit) {}
	BOT::Status OnTick() override;
};

class BuildStructure : public BOT::ActionNode {
	sc2::Tag location;

protected:
    sc2::UnitTypeID unitType;
    sc2::AbilityID abilityType;

public:
    BOT::Status PlaceBuilding(sc2::UnitTypeID unit, sc2::Point2D location, bool isExpansion); // Should be better, we should aim to use this instead of the method, I think
    BOT::Status PlaceBuilding(sc2::UnitTypeID unit, sc2::Tag loc);
	BuildStructure(sc2::UNIT_TYPEID unit) : unitType(unit), location(sc2::NullTag) {}
	BuildStructure(sc2::UNIT_TYPEID unit, sc2::Tag location) : unitType(unit), location(location) {}
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

class ShouldExpand : public BOT::ConditionNode {
    sc2::UNIT_TYPEID gasType;
public:
    ShouldExpand(sc2::UNIT_TYPEID gasType) : gasType(gasType) {}
    BOT::Status OnTick() override;
    int GetExpectedWorkers(sc2::UNIT_TYPEID vespene_building_type);
};

class BuildGas : public BuildStructure {

public:
	BuildGas(sc2::UnitTypeID unit) : BuildStructure(unit) {}
	BOT::Status OnTick() override;
};

class Expand: public BuildStructure{
public:
    Expand(sc2::UnitTypeID unit) : BuildStructure(unit) {}
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

class BuildAddon: public BOT::ActionNode{
    sc2::AbilityID abilityType;
    std::vector<sc2::UNIT_TYPEID> buildingTypes;
public:
    BOT::Status OnTick() override;
    BOT::Status TryBuildAddon(sc2::AbilityID ability_type_for_structure, sc2::Tag base_structure);
    BuildAddon(sc2::AbilityID ability, std::vector<sc2::UNIT_TYPEID> types) : abilityType(ability), buildingTypes(types) {}
};

class SimpleAttackMove: public BOT::ActionNode{
	BOT::Status OnTick() override;
};