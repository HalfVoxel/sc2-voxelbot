#pragma once
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"

class BuildingPlacement {
    void FindWallPlacements(sc2::Point3D startLocation_);
	std::map<sc2::UNIT_TYPEID, std::vector<sc2::Point2D>> locationQueues;
public:
	void OnGameStart();
	void OnStep();
	sc2::Point2D GetReasonablePlacement(sc2::UnitTypeID structureType);
    sc2::Point2D wallPlacement;
};

