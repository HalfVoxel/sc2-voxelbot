#pragma once
#include "SpendingManager.h"
#include "BuildOptimizerGenetic.h"
#include <vector>

struct BuildOrderTracker {
    std::set<const sc2::Unit*> knownUnits;
    std::map<sc2::UNIT_TYPEID, int> ignoreUnits;
    std::vector<const sc2::Unit*> buildOrderUnits;
    BuildOrder buildOrder;

    BuildOrderTracker () {}
    BuildOrderTracker (BuildOrder buildOrder);

    void setBuildOrder (BuildOrder buildOrder);
    void tweakBuildOrder (std::vector<bool> keepMask, BuildOrder buildOrder);
    void addExistingUnit(const sc2::Unit* unit);

    void ignoreUnit (sc2::UNIT_TYPEID type, int count);
    std::vector<bool> update(const std::vector<const sc2::Unit*>& ourUnits);
};

std::pair<int, std::vector<bool>> executeBuildOrder(const std::vector<const sc2::Unit*>& ourUnits, const BuildState& buildOrderStartingState, BuildOrderTracker& buildOrder, float currentMinerals, SpendingManager& spendingManager, bool& serialize);
void debugBuildOrderMasked(BuildState startingState, BuildOrder buildOrder, std::vector<bool> doneItems);
void debugBuildOrder(BuildState startingState, BuildOrder buildOrder, std::vector<bool> doneItems);

void debugBuildOrder(BuildState startingState, BuildOrder buildOrder, std::vector<float> doneTimes);
