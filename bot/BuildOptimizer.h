#pragma once
#include "sc2api/sc2_interfaces.h"

struct BuildOptimizer {
    void init(const sc2::ObservationInterface* observation);
    void calculate_build_order(std::vector<std::pair<sc2::UNIT_TYPEID, int>> start, std::vector<std::pair<sc2::UNIT_TYPEID, int>> target);
};