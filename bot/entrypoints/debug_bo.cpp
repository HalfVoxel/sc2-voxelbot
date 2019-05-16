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

string printBuildOrder(const BuildState& startingState, const BuildOrder& buildOrder) {
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
    initMappings();

    default_random_engine rnd(time(0));
    srand(time(0));

    BuildState startState;

    ifstream json("saved_buildorder_state.json");
    {
        cereal::JSONInputArchive archive(json);
        archive(startState);
    }
    // ofstream json2("saved_buildorder_gene.json");
    // {
    //     cereal::JSONOutputArchive archive(json2);
    //     BuildState state;
    //     archive(state);
    // }

    BuildOrder bo = {
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_NEXUS, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_CYBERNETICSCORE, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_SENTRY, false),
BuildOrderItem(UPGRADE_ID::WARPGATERESEARCH, false),

    };

    BuildOrder bo2 = {
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, true),
BuildOrderItem(UNIT_TYPEID::PROTOSS_NEXUS, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_CYBERNETICSCORE, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_SENTRY, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_TWILIGHTCOUNCIL, false),
BuildOrderItem(UNIT_TYPEID::PROTOSS_DARKSHRINE, false),

// BuildOrderItem(UPGRADE_ID::WARPGATERESEARCH, false),

    };

    mergeStrings(printBuildOrder(startState, bo), printBuildOrder(startState, bo2));

    auto f1 = calculateFitness(startState, bo);
    auto f2 = calculateFitness(startState, bo2);
    cout << f1.score() << " " << f2.score() << " Best build order? " << (f1 < f2 ? "bo2" : "bo1") << endl;
    cout << f1.time << " " << f2.time << endl;
}