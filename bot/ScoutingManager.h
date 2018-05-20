#pragma once
#include "Group.h"

class ScoutingManager{
    std::map<UnitGroup*, sc2::Point2D> scoutAssignments;
public:
    void OnStep();
    void ScoutDestroyed(UnitGroup* group);
    sc2::Point2D RequestScoutingPosition(UnitGroup* group);
};
