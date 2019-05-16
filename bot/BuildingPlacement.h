#pragma once
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include <libvoxelbot/utilities/influence.h>

class BuildingPlacement {
    void FindWallPlacements(sc2::Point3D startLocation_);
	std::map<sc2::UNIT_TYPEID, std::vector<sc2::Point2D>> locationQueues;
	std::vector<sc2::Point2D> lastKnownGoodPlacements;
public:
	void clearLastKnownGoodPlacements();
	void OnGameStart();
	void OnStep();
	sc2::Point2D GetReasonablePlacement(sc2::UnitTypeID structureType, sc2::ABILITY_ID abilityType = sc2::ABILITY_ID::INVALID, bool isWarping = false);
    sc2::Point2D wallPlacement;
};

