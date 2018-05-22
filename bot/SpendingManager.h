#pragma once
#include "sc2api/sc2_api.h"
#include <functional>

struct Cost {
    int minerals;
    int gas;
    int supply;
    sc2::UNIT_TYPEID unit_type;
};

double DefaultScore (sc2::UNIT_TYPEID unitType);
Cost CostOfUnit(sc2::UnitTypeID unitType);

struct SpendingManager {
private:
    std::vector<std::tuple<double, Cost, std::function<void()>>> actions;
public:
    void AddAction (double score, Cost cost, std::function<void()> action);
    void OnStep();
};