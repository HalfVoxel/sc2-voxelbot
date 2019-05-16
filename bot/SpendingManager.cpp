#include "SpendingManager.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include "utilities/mappings.h"
#include "Bot.h"

using namespace std;
using namespace sc2;

vector<pair<UNIT_TYPEID, double>> unitComposition = {
    { UNIT_TYPEID::TERRAN_MARINE, 5 },
    { UNIT_TYPEID::TERRAN_SIEGETANK, 1 },
    { UNIT_TYPEID::TERRAN_MEDIVAC, 1 },
    { UNIT_TYPEID::TERRAN_MARAUDER, 3 },
    { UNIT_TYPEID::TERRAN_CYCLONE, 0 },
    { UNIT_TYPEID::TERRAN_LIBERATOR, 1 },
    { UNIT_TYPEID::TERRAN_VIKINGFIGHTER, 1 },
    { UNIT_TYPEID::TERRAN_BANSHEE, 1 },
};

double SCVScore(UNIT_TYPEID unitType) {
    auto units = bot->Observation()->GetUnits(Unit::Alliance::Self);
    int ideal = 0;
    int assigned = 0;
    for (auto unit : units) {
        ideal += unit->ideal_harvesters;
        assigned += unit->assigned_harvesters;
    }

    if (assigned < ideal)
        return 10;
    if (assigned < ideal + 10)
        return 1.5;
    return 1;
}

double SpendingManager::GetUnitProportion(UNIT_TYPEID unitType) {
    double desiredMatching = 0;
    double totalCompositionWeight = 0;
    for (auto c : unitComposition) {
        totalCompositionWeight += c.second;
        if (c.first == unitType)
            desiredMatching += c.second;
    }

    return desiredMatching /= totalCompositionWeight + 0.0001;
}

double DefaultScore(UNIT_TYPEID unitType) {
    auto units = bot->Observation()->GetUnits(Unit::Alliance::Self);

    double matching = 0;
    double total = 0;
    for (auto unit : units) {
        for (auto c : unitComposition) {
            if (unit->unit_type == c.first) {
                total += 1;

                if (unit->unit_type == unitType) {
                    matching += 1;
                }
            }
        }
    }
    matching /= total + 0.0001;
    double desiredMatching = SpendingManager::GetUnitProportion(unitType);

    if (desiredMatching <= 0)
        return -10;

    // Score will be 1 exactly when we have the desired unit fraction.
    // It will go to a large value when the unit fraction we have is smaller than the desired one
    // and to 0 when the unit fraction we have of this unit approaches 100%.
    double score;
    if (matching < desiredMatching) {
        score = desiredMatching / (matching + 0.0001);
    } else {
        score = 1 - (matching - desiredMatching) / (1 - desiredMatching + 0.0001);
    }

    score = min(score, 5.0);
    return score;
}

Cost CostOfUpgrade(UpgradeID upgrade) {
    auto& data = getUpgradeData(upgrade);
    Cost result = {
        (int)data.mineral_cost,
        (int)data.vespene_cost,
        0,
        UNIT_TYPEID::INVALID
    };

    return result;
}

Cost CostOfUnit(UnitTypeID unit) {
    auto& unitData = getUnitData(unit);

    Cost result = {
        unitData.mineral_cost,
        unitData.vespene_cost,
        (int)unitData.food_required,
        unit
    };

    // Extensions are cumulative, fix that
    if (unit == UNIT_TYPEID::TERRAN_ORBITALCOMMAND || unit == UNIT_TYPEID::TERRAN_PLANETARYFORTRESS) {
        auto baseCost = CostOfUnit(UNIT_TYPEID::TERRAN_COMMANDCENTER);
        result.minerals -= baseCost.minerals;
        result.gas -= baseCost.gas;
    }

    return result;
}

bool HasTechFor(UnitTypeID unitType) {
    if (unitType == UNIT_TYPEID::INVALID) {
        return true;
    }

    auto required = getUnitData(unitType).tech_requirement;
    if (required == UNIT_TYPEID::INVALID) {
        return true;
    }

    for (auto unit : bot->ourUnits()) {
        if (unit->build_progress < 1) continue;

        if (unit->unit_type == required || simplifyUnitType(unit->unit_type) == required) {
            return true;
        }

        auto& aliases = getUnitData(unit->unit_type).tech_alias;
        if (contains(aliases, required)) {
            return true;
        }
    }

    return false;
}

void SpendingManager::AddAction(double score, Cost cost, std::function<void()> action, bool reserveResourcesOnly) {
    SpendingItem item;
    item.score = score;
    item.cost = cost;
    item.action = action;
    item.preparationTime = -1;
    item.preparationAction = nullptr;
    item.reserveResourcesOnly = reserveResourcesOnly;
    actions.push_back(item);
}

void SpendingManager::AddAction(double score, Cost cost, std::function<void()> action, float preparationTime, std::function<void()> preparationCallback) {
    SpendingItem item;
    item.score = score;
    item.cost = cost;
    item.action = action;
    item.preparationTime = preparationTime;
    item.preparationAction = preparationCallback;
    item.reserveResourcesOnly = false;
    actions.push_back(item);
}

vector<string> latestActions;

void SpendingManager::OnStep(const BuildState& buildState) {
    // Some actions take two frames to show up in the actual game (e.g. produce unit orders do not show up until after 2 frames)
    // so make sure we never execute actions more often than once every 2 frames
    if (agent->Observation()->GetGameLoop() - lastActionFrame < 2) {
        actions.clear();
        return;
    }

    sort(actions.begin(), actions.end(), [](const auto& a, const auto& b) -> bool {
        return b.score < a.score;
    });

    auto observation = bot->Observation();
    int totalMinerals = observation->GetMinerals();
    int totalGas = observation->GetVespene();
    int supply = observation->GetFoodCap() - observation->GetFoodUsed();

    stringstream ss;

    for (auto item : actions) {
        double score = item.score;
        Cost cost = item.cost;
        function<void()> callback = item.action;
        bool reserveResourcesOnly = item.reserveResourcesOnly;

        // ss << setw(4) << setprecision(2) << score << setw(22) << getUnitData(cost.unit_type).name << " min: " << setw(3) << cost.minerals << " gas: " << setw(3) << cost.gas << " food: " << cost.supply;
        ss << setw(4) << setprecision(2) << score << setw(22) << getUnitData(cost.unit_type).name;
        // Ignore any actions for which we don't have the required tech for yet

        bool hasTech = HasTechFor(cost.unit_type);
        if (!hasTech) {
            ss << " (no tech)" << endl;
            // continue;
        }

        totalMinerals -= cost.minerals;
        totalGas -= cost.gas;
        supply -= cost.supply;

        if (reserveResourcesOnly) {
            ss << " (reserved)" << endl;
            continue;
        }

        ss << endl;

        if ((cost.minerals == 0 || totalMinerals >= 0) && (cost.gas == 0 || totalGas >= 0) && (cost.supply == 0 || supply >= 0) && hasTech) {
            stringstream ss2;
            ss2 << setw(4) << setprecision(2) << score << setw(22) << getUnitData(cost.unit_type).name << " min: " << setw(3) << cost.minerals << " gas: " << setw(3) << cost.gas << " food: " << cost.supply;
            latestActions.push_back(ss2.str());
            if (latestActions.size() > 10)
                latestActions.erase(latestActions.begin());

            // Ok, we can use this ability
            callback();
            lastActionFrame = agent->Observation()->GetGameLoop();

            // Only process a single action per tick.
            // Technically it would work without a break, but then some bad things *may* happen, like a single SCV being assigned
            // two different build orders in the same turn, and that could lead to weird and hard to debug issues.
            break;
        }

        if (item.preparationAction != nullptr) {
            BuildState state = buildState;
            state.resources.minerals = totalMinerals + cost.minerals;
            state.resources.vespene = totalGas + cost.gas;
            // Note: no buildings require supply, and currently prep is only done for buildings. So we don't have to care that the supply might be inaccurate
            if (state.simulateBuildOrder({ item.cost.unit_type }, nullptr, false)) {
                if (buildState.time + item.preparationTime > state.time) {
                    // We estimate that we will be able to build this item soon
                    item.preparationAction();
                }
            }
        }

        // Note that it falls through here to the next action
        // however the minerals, gas and supply have already been reserved so later actions cannot use it.
        // This is important so that we are able to say build things that only cost minerals even while
        // blocked trying to build something that costs gas.
        // Or if we are supply blocked we should still be able to build buildings even though they are further
        // down the priority list.
    }

    bot->Debug()->DebugTextOut(ss.str(), bot->startLocation_);
    stringstream ss3;
    for (int i = latestActions.size() - 1; i >= 0; i--) {
        ss3 << latestActions[i] << endl;
    }
    bot->Debug()->DebugTextOut(ss3.str(), bot->startLocation_ + Point3D(10, 0, 0), Colors::Green);
    actions.clear();
}