#include <libvoxelbot/buildorder/optimizer.h>
#include "../build_order_helpers.h"
#include "sc2lib/sc2_lib.h"
#include "../behaviortree/TacticalNodes.h"
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/utilities/profiler.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <thread>
#include <fstream>
#include <sstream>
#include <random>
#include <iomanip>

using namespace std;
using namespace sc2;
using namespace BOT;

string timeStr(float time) {
    stringstream ss;
    int sec = (int)(fmod(time, 60.0f));
    ss << (int)(time / 60.0f) << ":" << (sec < 10 ? "0" : "") << sec;
    return ss.str();
}

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
        ss << timeStr(state.time) << " " << name;
        if (buildOrder.items[i].chronoBoosted) ss << " (chrono)";
        ss << std::setprecision(2) << std::fixed << " min: " << state.resources.minerals << "+" << (60*state.miningSpeed().mineralsPerSecond) << "/min";
        for (auto u : state.units) {
            if (u.type == UNIT_TYPEID::PROTOSS_PROBE) ss << getUnitData(u.type).name << "(" << u.availableUnits() << "/" << u.units << ") ";
        }
        ss << endl;
    });

    if (!success) ss << "FAILED " << "(food: " << state.foodAvailableInFuture() << ")" << endl;
    // ss << state.miningSpeed().mineralsPerSecond << " " << state.miningSpeed().vespenePerSecond << endl;
    ss << timeStr(state.time) << " Done" << endl;

    return ss.str();
}

static string mergeStrings (string s1, string s2) {
    stringstream ss1(s1);
    stringstream ss2(s2);
    stringstream merged;

    while(true) {
        string l1, l2;
        getline(ss1, l1);
        getline(ss2, l2);
        merged << setw(50) << std::left << l1 << " " << l2 << endl;

        if (!ss1 && !ss2) break;
    }
    return merged.str();
}

vector<BuildOrder> proBOEngine = {
    {
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
    },
    {
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_CYBERNETICSCORE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STARGATE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_FLEETBEACON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_CARRIER, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_CARRIER, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, false),
    },
    {
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_CYBERNETICSCORE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_NEXUS, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ROBOTICSBAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_COLOSSUS, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_COLOSSUS, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_COLOSSUS, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
    },
    {
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_NEXUS, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_CYBERNETICSCORE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STARGATE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ORACLE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_WARPPRISM, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_VOIDRAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_FLEETBEACON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_TWILIGHTCOUNCIL, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PHOENIX, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ROBOTICSBAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_DARKSHRINE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_CARRIER, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_OBSERVER, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_COLOSSUS, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_TEMPLARARCHIVE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_TEMPEST, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_HIGHTEMPLAR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_DARKTEMPLAR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ADEPT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_DISRUPTOR, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_SENTRY, true),
    },
    {
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_NEXUS, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_CYBERNETICSCORE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PROBE, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ASSIMILATOR, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_GATEWAY, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_PYLON, false),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_IMMORTAL, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER, true),
        BuildOrderItem(UNIT_TYPEID::PROTOSS_SENTRY, false),
    }
};

vector<pair<BuildOrderItem, int>> targetItemsFromBO(BuildOrder bo) {
    map<UNIT_TYPEID, int> counts;
    for (auto item : bo.items) {
        if (!item.isUnitType() || isArmy(item.rawType())) {
            counts[item.rawType()] += 1;
        }
    }

    vector<pair<BuildOrderItem, int>> targets;
    for (auto p : counts) {
        targets.push_back({BuildOrderItem(p.first), p.second});
    }
    return targets;
}

int main() {
    initMappings();

    default_random_engine rnd(time(0));
    srand(time(0));

    vector<int> its = { 256, 512, 1024, 2048 };

    for (int k = 1; k < 5; k++) {
        BuildState startState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 } });
        startState.resources.minerals = 50;
        startState.resources.vespene = 0;
        
        startState.race = Race::Protoss;
        startState.chronoInfo.addNexusWithEnergy(startState.time, 50);
        // Initial delay before harvesters start mining properly
        startState.makeUnitsBusy(UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::INVALID, 12);
        for (int i = 0; i < 12; i++) startState.addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, 4, UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::INVALID));

        BuildOptimizerParams params;
        params.iterations = 1024;
        Stopwatch w;
        auto boTuple = findBestBuildOrderGenetic(startState, { {BuildOrderItem(UNIT_TYPEID::PROTOSS_ZEALOT), 2*k}, {BuildOrderItem(UNIT_TYPEID::PROTOSS_STALKER), 2*k}, {BuildOrderItem(UNIT_TYPEID::PROTOSS_SENTRY), 2*k} }, nullptr, params);
        w.stop();
        cout << "Millis " << w.millis() << endl;
    }

    for (auto proBO : proBOEngine) {
        BuildState startState({ { UNIT_TYPEID::PROTOSS_NEXUS, 1 }, { UNIT_TYPEID::PROTOSS_PROBE, 12 } });
        startState.resources.minerals = 50;
        startState.resources.vespene = 0;
        
        startState.race = Race::Protoss;
        startState.chronoInfo.addNexusWithEnergy(startState.time, 50);
        // Initial delay before harvesters start mining properly
        startState.makeUnitsBusy(UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::INVALID, 12);
        for (int i = 0; i < 12; i++) startState.addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, 4, UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::INVALID));

        float proBOTime;
        {
            auto state = startState;
            state.simulateBuildOrder(proBO, nullptr, true);
            proBOTime = state.time;
        }
        BuildOptimizerParams params;
        vector<pair<int,float>> durations;
        for (auto iterations : its) {
            params.iterations = iterations;
            for (int i = 0; i < 0; i++) {
                auto boTuple = findBestBuildOrderGenetic(startState, targetItemsFromBO(proBO), nullptr, params);
                // cout << mergeStrings(printBuildOrder(startState, proBO), printBuildOrder(startState, boTuple.first)) << endl;

                // auto f1 = calculateFitness(startState, proBO);
                // auto f2 = calculateFitness(startState, boTuple.first);

                // cout << (f1 < f2 ? "Ours is better" : "Ours is worse") << endl;
                auto state = startState;
                state.simulateBuildOrder(boTuple.first, nullptr, true);
                durations.push_back({iterations, state.time});
            }
        }

        cout << "(" << proBOTime << ", [";
        for (auto t : durations) cout << "(" << t.first << ", " << t.second << ")" << ", ";
        cout << endl;

        for (auto u : targetItemsFromBO(proBO)) {
            cout << getUnitData(u.first.typeID()).name << " & " << u.second << endl;
        }
        cout << endl;
    }
}