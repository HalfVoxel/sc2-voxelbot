#pragma once
#include "sc2api/sc2_interfaces.h"

struct BuildOptimizer {
    void init();
    std::vector<sc2::UNIT_TYPEID> calculate_build_order(sc2::Race race, const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& start, const std::vector<std::pair<sc2::UNIT_TYPEID, int>>& target);
};

void unitTestBuildOptimizer(BuildOptimizer& optimizer);