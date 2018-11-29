#pragma once
#include "utilities/influence.h"

struct InfluenceManager {
	InfluenceMap pathing_grid;
	InfluenceMap pathing_cost;
	InfluenceMap placement_grid;
	InfluenceMap enemyDensity;
	InfluenceMap valueMap;
	InfluenceMap scoutingMap;
	InfluenceMap scanningMap;

	void Init();
	void OnStep();
};
