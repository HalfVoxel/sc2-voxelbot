#pragma once
#include <libvoxelbot/utilities/influence.h>

struct InfluenceManager {
	InfluenceMap pathing_grid;
	InfluenceMap pathing_cost;
	InfluenceMap pathing_cost_finite;
	InfluenceMap placement_grid;
	InfluenceMap enemyDensity;
	InfluenceMap valueMap;
	InfluenceMap scoutingMap;
	InfluenceMap scanningMap;
	InfluenceMap safeBuildingMap;
	InfluenceMap lastSeenMap;

	void Init();
	void OnStep();
};
