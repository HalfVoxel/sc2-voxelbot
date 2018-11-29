#pragma once
#include "BehaviorTree.h"
#include "sc2api/sc2_api.h"
#include <functional>
#include "../utilities/mappings.h"
#include "../bot.h"

class Build : public BOT::ActionNode {
	sc2::UnitTypeID unitType;
	std::function<double(sc2::UNIT_TYPEID)> score;

public:
	Build(sc2::UnitTypeID unit, std::function<double(sc2::UNIT_TYPEID)> score) : unitType(unit), score(score) {}
	BOT::Status OnTick() override;
};

class Research: public BOT::ActionNode{
    sc2::UpgradeID research;
    std::function<double(sc2::UPGRADE_ID)> score;
public:
    Research(sc2::UPGRADE_ID research, std::function<double(sc2::UPGRADE_ID)> score) : research(research), score(score) {}
    BOT::Status OnTick() override;
};

class Construct : public BOT::ActionNode {
	sc2::Tag location;
	std::function<double(sc2::UNIT_TYPEID)> score;

protected:
    sc2::UnitTypeID unitType;
    sc2::AbilityID abilityType;

public:
    BOT::Status PlaceBuilding(sc2::UnitTypeID unit, sc2::Point2D location, bool isExpansion); // Should be better, we should aim to use this instead of the method, I think
    BOT::Status PlaceBuilding(sc2::UnitTypeID unit, sc2::Tag loc);
	Construct(sc2::UNIT_TYPEID unit, std::function<double(sc2::UNIT_TYPEID)> score) : unitType(unit), location(sc2::NullTag), score(score) {}
	Construct(sc2::UNIT_TYPEID unit, sc2::Tag location, std::function<double(sc2::UNIT_TYPEID)> score) : unitType(unit), location(location), score(score) {}
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

class BuildGas : public Construct {

public:
	BuildGas(sc2::UnitTypeID unit, std::function<double(sc2::UNIT_TYPEID)> score) : Construct(unit, score) {}
	BOT::Status OnTick() override;
};

class Expand: public Construct{
public:
    Expand(sc2::UnitTypeID unit, std::function<double(sc2::UNIT_TYPEID)> score) : Construct(unit, score) {}
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

class Addon: public BOT::ActionNode{
    sc2::AbilityID abilityType;
    std::vector<sc2::UNIT_TYPEID> buildingTypes;
	std::function<double(sc2::UNIT_TYPEID)> score;
public:
    BOT::Status OnTick() override;
    BOT::Status TryBuildAddon(sc2::AbilityID ability_type_for_structure, sc2::Tag base_structure);
    Addon(sc2::AbilityID ability, std::vector<sc2::UNIT_TYPEID> types, std::function<double(sc2::UNIT_TYPEID)> score) : abilityType(ability), buildingTypes(types), score(score) {}
};

class HasUpgrade : public BOT::ActionNode {
    sc2::UpgradeID upgrade;
    sc2::AbilityID upgradeBuild;
public:
    HasUpgrade(sc2::UpgradeID upgrade);
    BOT::Status OnTick() override;
};





