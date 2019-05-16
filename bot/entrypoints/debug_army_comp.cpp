#include <libvoxelbot/utilities/mappings.h>
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/buildorder/optimizer.h>
#include <libvoxelbot/buildorder/build_time_estimator.h>
#include <libvoxelbot/utilities/profiler.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cereal/cereal.hpp>
#include <libvoxelbot/utilities/cereal_json.h>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <libvoxelbot/common/unit_lists.h>
#include <libvoxelbot/utilities/mappings.h>
#include <libvoxelbot/utilities/build_state_serialization.h>
#include "../build_order_helpers.h"

using namespace std;
using namespace sc2;

static string printBuildOrder(const BuildState& startingState, const BuildOrder& buildOrder) {
    BuildState state = startingState;
    stringstream ss;
    for (auto u : startingState.units) {
        ss << getUnitData(u.type).name << "(" << u.availableUnits() << "/" << u.units << ") ";
    }
    ss << endl;
    // ss << "Time: " << (int)round(buildOrderTime) << endl;
    bool success = state.simulateBuildOrder(buildOrder, [&](int i) {
        string name = buildOrder[i].isUnitType() ? getUnitData(buildOrder[i].typeID()).name : UpgradeIDToName(buildOrder[i].upgradeID());
        int sec = (int)(fmod(state.time, 60.0f));
        ss << (int)(state.time / 60.0f) << ":" << (sec < 10 ? "0" : "") << sec << " " << name;
        if (buildOrder.items[i].chronoBoosted) ss << " (chrono)";
        ss << std::setprecision(2) << std::fixed << " min: " << state.resources.minerals << "+" << (60*state.miningSpeed().mineralsPerSecond) << "/min";
        for (auto u : state.units) {
            if (u.type == UNIT_TYPEID::PROTOSS_PROBE) ss << getUnitData(u.type).name << "(" << u.availableUnits() << "/" << u.units << ") ";
        }
        ss << endl;
    });

    if (!success) ss << "FAILED " << "(food: " << state.foodAvailableInFuture() << ")" << endl;
    ss << state.miningSpeed().mineralsPerSecond << " " << state.miningSpeed().vespenePerSecond << endl;

    return ss.str();
}

static void mergeStrings (string s1, string s2) {
    stringstream ss1(s1);
    stringstream ss2(s2);

    while(true) {
        string l1, l2;
        getline(ss1, l1);
        getline(ss2, l2);
        cout << setw(50) << std::left << l1 << " " << l2 << endl;

        if (!ss1 && !ss2) break;
    }
}

int main() {
#if !DISABLE_PYTHON
    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        import os
        sys.path.append("bot/python")
        os.environ["MPLBACKEND"] = "TkAgg"
    )");
#endif

    initMappings();

    default_random_engine rnd(time(0));
    srand(time(0));

    BuildState startState;

    ifstream json("saved_buildorder_state.json");
    {
        cereal::JSONInputArchive archive(json);
        archive(startState);
    }

    startState = BuildState({
        { UNIT_TYPEID::PROTOSS_NEXUS, 2 },
        { UNIT_TYPEID::PROTOSS_PROBE, 36 },
        { UNIT_TYPEID::PROTOSS_ASSIMILATOR, 2 },
        { UNIT_TYPEID::PROTOSS_ZEALOT, 12 },
        { UNIT_TYPEID::PROTOSS_ADEPT, 5 },
    });

    CombatPredictor combatPredictor;
    combatPredictor.init();
    BuildOptimizerNN buildTimePredictor;
    buildTimePredictor.init();

    CombatState combatState;
    // for (auto u : startState.units) {
    //     if (isArmy(u.type)) {
    //         for (int i = 0; i < u.units; i++) combatState.units.push_back(makeUnit(2, u.type));
    //     }
    // }

    vector<CombatUnit> enemyUnits = {
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_SIEGETANKSIEGED),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_SIEGETANKSIEGED),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_SIEGETANKSIEGED),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        // // makeUnit(1, UNIT_TYPEID::TERRAN_MEDIVAC),
        // makeUnit(1, UNIT_TYPEID::TERRAN_MEDIVAC),


        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
        makeUnit(1, UNIT_TYPEID::PROTOSS_COLOSSUS),
        // makeUnit(1, UNIT_TYPEID::PROTOSS_DARKTEMPLAR),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_ROACH),
        makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
        makeUnit(1, UNIT_TYPEID::ZERG_HYDRALISK),
    };

    for (auto u : enemyUnits) {
        combatState.units.push_back(u);
    }

    ArmyComposition bestComposition;

    BuildState tmpState = startState;

    for (int k = 0; k < 2; k++) {
        auto futureState = tmpState;
        futureState.simulate(futureState.time + 40);
        futureState.resources = tmpState.resources;
        
        CompositionSearchSettings settings(combatPredictor, getAvailableUnitsForRace(Race::Protoss, UnitCategory::ArmyCompositionOptions), &buildTimePredictor);
        settings.availableTime = k == 0 ? 4 * 60 : 6 * 60;
        auto bestCounter = findBestCompositionGenetic(combatState, settings, &futureState, nullptr);

        bestComposition.combine(bestCounter);

        for (auto c : bestCounter.unitCounts) {
            cout << c.second << "x\t" << getUnitData(c.first).name << endl;
            // for (int i = 0; i < c.second; i++) combatState.units.push_back(makeUnit(2, c.first));
        }

        vector<pair<int, int>> startingUnitsNN;
        vector<pair<int, int>> targetUnitsNN;

        map<int, int> unitCounts;
        vector<pair<BuildOrderItem, int>> targetUnits;
        // for (auto u : combatState.units) {
        //     if (u.owner == 2) unitCounts[(int)u.type]++;
        // }
        for (auto u : tmpState.units) {
            unitCounts[(int)u.type] += u.units;
        }
        for (auto c : bestCounter.unitCounts) {
            unitCounts[(int)c.first] += c.second;
        }

        for (auto p : unitCounts) {
            targetUnitsNN.push_back({ p.first, p.second });
            targetUnits.push_back({BuildOrderItem((UNIT_TYPEID)p.first), p.second});
        }

        for (auto u : futureState.units) {
            startingUnitsNN.push_back({(int)u.type, u.units});
        }
        
        for (auto u : bestCounter.upgrades) {
            cout << UpgradeIDToName(u) << endl;
        }

        vector<vector<float>> timesToProduceUnits = buildTimePredictor.predictTimeToBuild(startingUnitsNN, tmpState.resources, { targetUnitsNN });

        auto tmpCombatState = combatState;
        for (auto u : bestCounter.unitCounts) {
            for (int i = 0; i < u.second; i++) tmpCombatState.units.push_back(makeUnit(2, u.first));
        }
        cout << "Mineral score: " << combatPredictor.mineralScore(tmpCombatState, combatPredictor.predict_engage(tmpCombatState, {}), 2, timesToProduceUnits[0], {}) << endl;
        cout << "Mineral score (fixed): " << combatPredictor.mineralScoreFixedTime(tmpCombatState, combatPredictor.predict_engage(tmpCombatState, {}), 2, timesToProduceUnits[0], {}) << endl;
        logRecordings(tmpCombatState, combatPredictor, 0, "debug_army_comp");

        auto buildOrder = findBestBuildOrderGenetic(tmpState, targetUnits, nullptr).first;
        cout << printBuildOrder(tmpState, buildOrder) << endl;

        tmpState.simulateBuildOrder(buildOrder, nullptr, false);
        // cout << "BO?" << endl;
    }

    cout << "Final counter " << startState.time << endl;
    {
        for (auto c : bestComposition.unitCounts) {
            cout << c.second << "x\t" << getUnitData(c.first).name << endl;
            // for (int i = 0; i < c.second; i++) combatState.units.push_back(makeUnit(2, c.first));
        }

        map<int, int> unitCounts;
        vector<pair<BuildOrderItem, int>> targetUnits;
        for (auto c : bestComposition.unitCounts) {
            unitCounts[(int)c.first] += c.second;
        }

        for (auto p : unitCounts) {
            targetUnits.push_back({BuildOrderItem((UNIT_TYPEID)p.first), p.second});
        }

        auto buildOrder = findBestBuildOrderGenetic(startState, targetUnits, nullptr).first;
        // startState.simulateBuildOrder(buildOrder, nullptr, false);
        cout << printBuildOrder(startState, buildOrder) << endl;
    }
}
