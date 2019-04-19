#pragma once
#include "sc2api/sc2_api.h"
#include <functional>
#include "BuildOptimizerGenetic.h"

struct Cost {
    int minerals;
    int gas;
    int supply;
    sc2::UNIT_TYPEID unit_type;
};

double SCVScore (sc2::UNIT_TYPEID unitType);
double DefaultScore (sc2::UNIT_TYPEID unitType);
Cost CostOfUnit(sc2::UnitTypeID unitType);
Cost CostOfUpgrade(sc2::UpgradeID upgrade);

struct SpendingItem {
    float score = 0;
    Cost cost;
    std::function<void()> action = nullptr;
    bool reserveResourcesOnly = false;
    float preparationTime = 0;
    std::function<void()> preparationAction = nullptr;
};

struct SpendingManager {
private:
    std::vector<SpendingItem> actions;
    int lastActionFrame = -1000;
public:
    static double GetUnitProportion(sc2::UNIT_TYPEID unitType);
    void AddAction (double score, Cost cost, std::function<void()> action, bool reserveResourcesOnly = false);
    void AddAction (double score, Cost cost, std::function<void()> action, float preparationTime, std::function<void()> preparationCallback);
    void OnStep(const BuildState& buildState);
};
