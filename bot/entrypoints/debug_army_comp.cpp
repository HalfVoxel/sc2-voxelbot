#include "../utilities/mappings.h"
#include "../utilities/predicates.h"
#include "../BuildOptimizerGenetic.h"
#include "../build_optimizer_nn.h"
#include "../utilities/profiler.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cereal/cereal.hpp>
#include "../utilities/cereal_json.h"
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include "../unit_lists.h"
#include "../utilities/mappings.h"
#include "../utilities/build_state_serialization.h"
#include "../build_order_helpers.h"

using namespace std;
using namespace sc2;

void mergeStrings (string s1, string s2) {
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

    auto futureState = startState;
    futureState.simulate(futureState.time + 40);
    futureState.resources = startState.resources;

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
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARINE),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MARAUDER),
        makeUnit(1, UNIT_TYPEID::TERRAN_MEDIVAC),
        makeUnit(1, UNIT_TYPEID::TERRAN_MEDIVAC),


        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),

        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),

        // makeUnit(2, UNIT_TYPEID::PROTOSS_SENTRY),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_SENTRY),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_SENTRY),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_SENTRY),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_SENTRY),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_SENTRY),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_SENTRY),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_SENTRY),

        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ZEALOT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ADEPT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ADEPT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ADEPT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ADEPT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ADEPT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ADEPT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_ADEPT),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_IMMORTAL),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_IMMORTAL),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_IMMORTAL),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_STALKER),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
        // makeUnit(2, UNIT_TYPEID::PROTOSS_COLOSSUS),
    };

    for (auto u : enemyUnits) {
        combatState.units.push_back(u);
    }

    CombatSettings settings;
    auto bestCounter = findBestCompositionGenetic(combatPredictor, getAvailableUnitsForRace(Race::Protoss, UnitCategory::ArmyCompositionOptions), combatState, &buildTimePredictor, &futureState, nullptr);
    // auto bestCounter = findBestCompositionGenetic(combatPredictor, getAvailableUnitsForRace(Race::Protoss, UnitCategory::ArmyCompositionOptions), combatState, nullptr, &futureState, nullptr);

    for (auto c : bestCounter.unitCounts) {
        cout << c.second << "x\t" << getUnitData(c.first).name << endl;
        // for (int i = 0; i < c.second; i++) combatState.units.push_back(makeUnit(2, c.first));
    }
    for (auto u : bestCounter.upgrades) {
        cout << UpgradeIDToName(u) << endl;
    }

    cout << "Mineral score: " << combatPredictor.mineralScore(combatState, combatPredictor.predict_engage(combatState, {}), 2, { 0, 0, 0 }, {}) << endl;

    logRecordings(combatState, combatPredictor, 0, "debug_army_comp");
}