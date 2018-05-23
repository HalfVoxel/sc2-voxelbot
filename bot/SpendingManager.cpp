#include "SpendingManager.h"
#include <algorithm>
#include "bot.h"
#include "Mappings.h"
#include <iostream>
#include <map>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sc2;

vector<pair<UNIT_TYPEID, double>> unitComposition = {
    { UNIT_TYPEID::TERRAN_MARINE, 5 },
    { UNIT_TYPEID::TERRAN_SIEGETANK, 1 },
    { UNIT_TYPEID::TERRAN_MEDIVAC, 1 },
    { UNIT_TYPEID::TERRAN_MARAUDER, 3 },
    { UNIT_TYPEID::TERRAN_CYCLONE, 2 },
    { UNIT_TYPEID::TERRAN_LIBERATOR, 1 },
    { UNIT_TYPEID::TERRAN_VIKINGFIGHTER, 1 },
    { UNIT_TYPEID::TERRAN_BANSHEE, 1 },
};

double SCVScore (UNIT_TYPEID unitType) {
    auto units = bot.Observation()->GetUnits(Unit::Alliance::Self);
    int ideal = 0;
    int assigned = 0;
    for (auto unit : units) {
        ideal += unit->ideal_harvesters;
        assigned += unit->assigned_harvesters;
    }

    if (assigned < ideal) return 10;
    if (assigned < ideal + 10) return 1.5;
    return 1;
}

double SpendingManager::GetUnitProportion(UNIT_TYPEID unitType) {
    double desiredMatching = 0;
    double totalCompositionWeight = 0;
    for (auto c : unitComposition) {
        totalCompositionWeight += c.second;
        if (c.first == unitType) desiredMatching += c.second;
    }

    return desiredMatching /= totalCompositionWeight + 0.0001;
}

double DefaultScore (UNIT_TYPEID unitType) {
    auto units = bot.Observation()->GetUnits(Unit::Alliance::Self);

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

Cost CostOfUnit(UnitTypeID unit) {
    auto& unitData = bot.Observation()->GetUnitTypeData()[unit];

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

    auto unitData = bot.Observation()->GetUnitTypeData();
    auto required = unitData[unitType].tech_requirement;
    if (required == UNIT_TYPEID::INVALID) {
        return true;
    }

    for (auto unit : bot.Observation()->GetUnits(Unit::Alliance::Self)) {
        if (unit->unit_type == required || simplifyUnitType(unit->unit_type) == required) {
            return true;
        }

        auto& aliases = unitData[unit->unit_type].tech_alias;
        if (find(aliases.begin(), aliases.end(), required) != aliases.end()) {
            return true;
        }
    }

    return false;
}

void SpendingManager::AddAction(double score, Cost cost, std::function<void()> action) {
    actions.push_back(make_tuple(score, cost, action));
}

vector<string> latestActions;

void SpendingManager::OnStep() {
    sort(actions.begin(), actions.end(), [](const auto& a, const auto& b) -> bool {
        return get<0>(b) < get<0>(a);
    });

    auto observation = bot.Observation();
    int totalMinerals = observation->GetMinerals();
    int totalGas = observation->GetVespene();
    int supply = observation->GetFoodCap() - observation->GetFoodUsed();

    stringstream ss;

    for (auto action : actions) {
        double score;
        Cost cost;
        function<void()> callback;
        tie(score, cost, callback) = action;

        ss << setw(4) << setprecision(2) << score << setw(22) << UnitTypeToName(cost.unit_type) << " min: " << setw(3) << cost.minerals << " gas: " << setw(3) << cost.gas << " food: " << cost.supply;
        // Ignore any actions for which we don't have the required tech for yet
        if (!HasTechFor(cost.unit_type)) {
            ss << " (ignored due to tech)" << endl;
            continue;
        } else {
            ss << endl;
        }

        totalMinerals -= cost.minerals;
        totalGas -= cost.gas;
        supply -= cost.supply;

        if ((cost.minerals == 0 || totalMinerals >= 0) && (cost.gas == 0 || totalGas >= 0) && (cost.supply == 0 || supply >= 0)) {
            stringstream ss2;
            ss2 << setw(4) << setprecision(2) << score << setw(22) << UnitTypeToName(cost.unit_type) << " min: " << setw(3) << cost.minerals << " gas: " << setw(3) << cost.gas << " food: " << cost.supply;
            latestActions.push_back(ss2.str());
            if (latestActions.size() > 10) latestActions.erase(latestActions.begin());

            // Ok, we can use this ability
            callback();

            // Only process a single action per tick.
            // Technically it would work without a break, but then some bad things *may* happen, like a single SCV being assigned
            // two different build orders in the same turn, and that could lead to weird and hard to debug issues.
            break;
        }

        // Note that it falls through here to the next action
        // however the minerals, gas and supply have already been reserved so later actions cannot use it.
        // This is important so that we are able to say build things that only cost minerals even while
        // blocked trying to build something that costs gas.
        // Or if we are supply blocked we should still be able to build buildings even though they are further
        // down the priority list.
    }

    bot.Debug()->DebugTextOut(ss.str(), bot.startLocation_);
    stringstream ss3;
    for (int i = latestActions.size() - 1; i >= 0; i--) {
        ss3 << latestActions[i] << endl;
    }
    bot.Debug()->DebugTextOut(ss3.str(), bot.startLocation_ + Point3D(10, 0, 0), Colors::Green);
    actions.clear();
}