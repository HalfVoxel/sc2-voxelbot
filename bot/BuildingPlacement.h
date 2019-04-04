#pragma once
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "utilities/influence.h"

// Where it is safe to build buildings
extern InfluenceMap safeBuildingMap;

class BuildingPlacement {
    void FindWallPlacements(sc2::Point3D startLocation_);
	std::map<sc2::UNIT_TYPEID, std::vector<sc2::Point2D>> locationQueues;
public:
	void OnGameStart();
	void OnStep();
	sc2::Point2D GetReasonablePlacement(sc2::UnitTypeID structureType, sc2::ABILITY_ID abilityType = sc2::ABILITY_ID::INVALID);
    sc2::Point2D wallPlacement;
};

