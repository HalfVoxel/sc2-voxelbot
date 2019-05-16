#pragma once
#include "SpendingManager.h"
#include <libvoxelbot/buildorder/optimizer.h>
#include <vector>

std::pair<int, std::vector<bool>> executeBuildOrder(const sc2::ObservationInterface* observation, const std::vector<const sc2::Unit*>& ourUnits, const BuildState& buildOrderStartingState, BuildOrderTracker& buildOrder, float currentMinerals, SpendingManager& spendingManager, bool& serialize);
void debugBuildOrderMasked(BuildState startingState, BuildOrder buildOrder, std::vector<bool> doneItems);
void debugBuildOrder(BuildState startingState, BuildOrder buildOrder, std::vector<bool> doneItems);

void debugBuildOrder(BuildState startingState, BuildOrder buildOrder, std::vector<float> doneTimes);
