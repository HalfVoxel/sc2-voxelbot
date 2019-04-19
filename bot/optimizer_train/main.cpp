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
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

using namespace sc2;
using namespace std;

static vector<UNIT_TYPEID> unitTypesTerran2 = {
    UNIT_TYPEID::TERRAN_ARMORY,
    UNIT_TYPEID::TERRAN_BANSHEE,
    UNIT_TYPEID::TERRAN_BARRACKS,
    // UNIT_TYPEID::TERRAN_BARRACKSREACTOR,
    // UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
    UNIT_TYPEID::TERRAN_BATTLECRUISER,
    UNIT_TYPEID::TERRAN_BUNKER,
    UNIT_TYPEID::TERRAN_COMMANDCENTER,
    UNIT_TYPEID::TERRAN_CYCLONE,
    UNIT_TYPEID::TERRAN_ENGINEERINGBAY,
    UNIT_TYPEID::TERRAN_FACTORY,
    // UNIT_TYPEID::TERRAN_FACTORYREACTOR,
    // UNIT_TYPEID::TERRAN_FACTORYTECHLAB,
    UNIT_TYPEID::TERRAN_FUSIONCORE,
    UNIT_TYPEID::TERRAN_GHOST,
    UNIT_TYPEID::TERRAN_GHOSTACADEMY,
    UNIT_TYPEID::TERRAN_HELLION,
    UNIT_TYPEID::TERRAN_HELLIONTANK,
    UNIT_TYPEID::TERRAN_LIBERATOR,
    UNIT_TYPEID::TERRAN_MARAUDER,
    UNIT_TYPEID::TERRAN_MARINE,
    UNIT_TYPEID::TERRAN_MEDIVAC,
    UNIT_TYPEID::TERRAN_MISSILETURRET,
    // UNIT_TYPEID::TERRAN_MULE,
    UNIT_TYPEID::TERRAN_ORBITALCOMMAND,
    UNIT_TYPEID::TERRAN_PLANETARYFORTRESS,
    UNIT_TYPEID::TERRAN_RAVEN,
    UNIT_TYPEID::TERRAN_REAPER,
    UNIT_TYPEID::TERRAN_REFINERY,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SENSORTOWER,
    UNIT_TYPEID::TERRAN_SIEGETANK,
    UNIT_TYPEID::TERRAN_STARPORT,
    // UNIT_TYPEID::TERRAN_STARPORTREACTOR,
    // UNIT_TYPEID::TERRAN_STARPORTTECHLAB,
    UNIT_TYPEID::TERRAN_SUPPLYDEPOT,
    UNIT_TYPEID::TERRAN_THOR,
    UNIT_TYPEID::TERRAN_VIKINGFIGHTER,
    UNIT_TYPEID::TERRAN_WIDOWMINE,
};

static vector<UNIT_TYPEID> unitTypesTerran4 = {
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_THOR,
};

static vector<UNIT_TYPEID> unitTypesTerran3 = {
    // UNIT_TYPEID::TERRAN_ARMORY,
    UNIT_TYPEID::TERRAN_BANSHEE,
    // UNIT_TYPEID::TERRAN_BARRACKS,
    // UNIT_TYPEID::TERRAN_BARRACKSREACTOR,
    // UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
    UNIT_TYPEID::TERRAN_BATTLECRUISER,
    // UNIT_TYPEID::TERRAN_BUNKER,
    // UNIT_TYPEID::TERRAN_COMMANDCENTER,
    UNIT_TYPEID::TERRAN_CYCLONE,
    // UNIT_TYPEID::TERRAN_ENGINEERINGBAY,
    // UNIT_TYPEID::TERRAN_FACTORY,
    // UNIT_TYPEID::TERRAN_FACTORYREACTOR,
    // UNIT_TYPEID::TERRAN_FACTORYTECHLAB,
    // UNIT_TYPEID::TERRAN_FUSIONCORE,
    UNIT_TYPEID::TERRAN_GHOST,
    // UNIT_TYPEID::TERRAN_GHOSTACADEMY,
    UNIT_TYPEID::TERRAN_HELLION,
    UNIT_TYPEID::TERRAN_HELLIONTANK,
    UNIT_TYPEID::TERRAN_LIBERATOR,
    UNIT_TYPEID::TERRAN_MARAUDER,
    UNIT_TYPEID::TERRAN_MARINE,
    UNIT_TYPEID::TERRAN_MEDIVAC,
    UNIT_TYPEID::TERRAN_MISSILETURRET,
    // UNIT_TYPEID::TERRAN_MULE,
    // UNIT_TYPEID::TERRAN_ORBITALCOMMAND,
    // UNIT_TYPEID::TERRAN_PLANETARYFORTRESS,
    UNIT_TYPEID::TERRAN_RAVEN,
    UNIT_TYPEID::TERRAN_REAPER,
    // UNIT_TYPEID::TERRAN_REFINERY,
    UNIT_TYPEID::TERRAN_SCV,
    // UNIT_TYPEID::TERRAN_SENSORTOWER,
    UNIT_TYPEID::TERRAN_SIEGETANK,
    // UNIT_TYPEID::TERRAN_STARPORT,
    // UNIT_TYPEID::TERRAN_STARPORTREACTOR,
    // UNIT_TYPEID::TERRAN_STARPORTTECHLAB,
    // UNIT_TYPEID::TERRAN_SUPPLYDEPOT,
    UNIT_TYPEID::TERRAN_THOR,
    UNIT_TYPEID::TERRAN_VIKINGFIGHTER,
    UNIT_TYPEID::TERRAN_WIDOWMINE,
};



vector<UNIT_TYPEID> unitTypesProtossMilitary = {
    UNIT_TYPEID::PROTOSS_ADEPT,
    // UNIT_TYPEID::PROTOSS_ADEPTPHASESHIFT,
    // UNIT_TYPEID::PROTOSS_ARCHON, // TODO: Special case creation rule
    // UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_CARRIER,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    // UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
    // UNIT_TYPEID::PROTOSS_DARKSHRINE,
    UNIT_TYPEID::PROTOSS_DARKTEMPLAR,
    UNIT_TYPEID::PROTOSS_DISRUPTOR,
    // UNIT_TYPEID::PROTOSS_DISRUPTORPHASED,
    // UNIT_TYPEID::PROTOSS_FLEETBEACON,
    // UNIT_TYPEID::PROTOSS_FORGE,
    // UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_HIGHTEMPLAR,
    UNIT_TYPEID::PROTOSS_IMMORTAL,
    // UNIT_TYPEID::PROTOSS_INTERCEPTOR,
    // UNIT_TYPEID::PROTOSS_MOTHERSHIP, // TODO: Mothership cannot be created for some reason (no unit has the required ability)
    // UNIT_TYPEID::PROTOSS_MOTHERSHIPCORE,
    // UNIT_TYPEID::PROTOSS_NEXUS,
    UNIT_TYPEID::PROTOSS_OBSERVER,
    UNIT_TYPEID::PROTOSS_ORACLE,
    // UNIT_TYPEID::PROTOSS_ORACLESTASISTRAP,
    // UNIT_TYPEID::PROTOSS_PHOENIX,
    // UNIT_TYPEID::PROTOSS_PHOTONCANNON,
    UNIT_TYPEID::PROTOSS_PROBE,
    // UNIT_TYPEID::PROTOSS_PYLON,
    // UNIT_TYPEID::PROTOSS_PYLONOVERCHARGED,
    // UNIT_TYPEID::PROTOSS_ROBOTICSBAY,
    // UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
    UNIT_TYPEID::PROTOSS_SENTRY,
    // UNIT_TYPEID::PROTOSS_SHIELDBATTERY,
    UNIT_TYPEID::PROTOSS_STALKER,
    // UNIT_TYPEID::PROTOSS_STARGATE,
    UNIT_TYPEID::PROTOSS_TEMPEST,
    // UNIT_TYPEID::PROTOSS_TEMPLARARCHIVE,
    // UNIT_TYPEID::PROTOSS_TWILIGHTCOUNCIL,
    UNIT_TYPEID::PROTOSS_VOIDRAY,
    // UNIT_TYPEID::PROTOSS_WARPGATE,
    UNIT_TYPEID::PROTOSS_WARPPRISM,
    // UNIT_TYPEID::PROTOSS_WARPPRISMPHASING,
    UNIT_TYPEID::PROTOSS_ZEALOT,
};

vector<UNIT_TYPEID> unitTypesProtossAll = {
    UNIT_TYPEID::PROTOSS_ADEPT,
    // UNIT_TYPEID::PROTOSS_ADEPTPHASESHIFT,
    // UNIT_TYPEID::PROTOSS_ARCHON, // TODO: Special case creation rule
    UNIT_TYPEID::PROTOSS_ASSIMILATOR,
    UNIT_TYPEID::PROTOSS_CARRIER,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    UNIT_TYPEID::PROTOSS_CYBERNETICSCORE,
    UNIT_TYPEID::PROTOSS_DARKSHRINE,
    UNIT_TYPEID::PROTOSS_DARKTEMPLAR,
    UNIT_TYPEID::PROTOSS_DISRUPTOR,
    // UNIT_TYPEID::PROTOSS_DISRUPTORPHASED,
    UNIT_TYPEID::PROTOSS_FLEETBEACON,
    UNIT_TYPEID::PROTOSS_FORGE,
    UNIT_TYPEID::PROTOSS_GATEWAY,
    UNIT_TYPEID::PROTOSS_HIGHTEMPLAR,
    UNIT_TYPEID::PROTOSS_IMMORTAL,
    // UNIT_TYPEID::PROTOSS_INTERCEPTOR,
    // UNIT_TYPEID::PROTOSS_MOTHERSHIP,
    // UNIT_TYPEID::PROTOSS_MOTHERSHIPCORE,
    UNIT_TYPEID::PROTOSS_NEXUS,
    UNIT_TYPEID::PROTOSS_OBSERVER,
    UNIT_TYPEID::PROTOSS_ORACLE,
    // UNIT_TYPEID::PROTOSS_ORACLESTASISTRAP,
    UNIT_TYPEID::PROTOSS_PHOENIX,
    UNIT_TYPEID::PROTOSS_PHOTONCANNON,
    UNIT_TYPEID::PROTOSS_PROBE,
    UNIT_TYPEID::PROTOSS_PYLON,
    // UNIT_TYPEID::PROTOSS_PYLONOVERCHARGED,
    UNIT_TYPEID::PROTOSS_ROBOTICSBAY,
    UNIT_TYPEID::PROTOSS_ROBOTICSFACILITY,
    UNIT_TYPEID::PROTOSS_SENTRY,
    UNIT_TYPEID::PROTOSS_SHIELDBATTERY,
    UNIT_TYPEID::PROTOSS_STALKER,
    UNIT_TYPEID::PROTOSS_STARGATE,
    UNIT_TYPEID::PROTOSS_TEMPEST,
    UNIT_TYPEID::PROTOSS_TEMPLARARCHIVE,
    UNIT_TYPEID::PROTOSS_TWILIGHTCOUNCIL,
    UNIT_TYPEID::PROTOSS_VOIDRAY,
    UNIT_TYPEID::PROTOSS_WARPGATE,
    UNIT_TYPEID::PROTOSS_WARPPRISM,
    // UNIT_TYPEID::PROTOSS_WARPPRISMPHASING,
    UNIT_TYPEID::PROTOSS_ZEALOT,
};

vector<pair<UNIT_TYPEID, int>> sampleTargetUnitConfig (default_random_engine& rnd) {
    // auto& pool = guaranteeFood ? unitTypesTerran2 : unitTypesTerran3;
    auto& pool = unitTypesProtossMilitary;
    Race race = Race::Protoss;

    exponential_distribution<float> numUnitsDist;
    if (bernoulli_distribution(0.2)(rnd)) {
        numUnitsDist = exponential_distribution<float>(1.0/1);
    } else if (bernoulli_distribution(0.05)(rnd)) {
        numUnitsDist = exponential_distribution<float>(1.0/10.0);
    } else {
        numUnitsDist = exponential_distribution<float>(1.0/5.0);
    }
    int numUnits = min((int)round(numUnitsDist(rnd)+1), (int)pool.size());
    // if (!guaranteeFood) numUnits = numUnits;
    vector<pair<UNIT_TYPEID, int>> result;
    for (int i = 0; i < numUnits; i++) {
        uniform_int_distribution<int> typeDist(0, pool.size()-1);
        UNIT_TYPEID type = pool[typeDist(rnd)];
        exponential_distribution<double> countDist;

        if (bernoulli_distribution(0.05)(rnd)) {
            countDist = exponential_distribution<double>(1.0/20.0);
        } else {
            countDist = exponential_distribution<double>(1.0/2.0);
        }

        int cnt = (int)round(countDist(rnd));
        if (cnt > 0) {
            result.push_back(make_pair(type, min(100, cnt)));
        }
    }

    return result;
}

vector<pair<UNIT_TYPEID, int>> sampleUnitConfig (default_random_engine& rnd, bool guaranteeFood) {
    // auto& pool = guaranteeFood ? unitTypesTerran2 : unitTypesTerran3;
    auto& pool = guaranteeFood ? unitTypesProtossAll : unitTypesProtossMilitary;
    Race race = Race::Protoss;

    exponential_distribution<float> numUnitsDist;
    if (bernoulli_distribution(0.1)(rnd) && !guaranteeFood) {
        numUnitsDist = exponential_distribution<float>(1.0/1);
    } else if (bernoulli_distribution(0.05)(rnd)) {
        numUnitsDist = exponential_distribution<float>(1.0/10.0);
    } else {
        numUnitsDist = exponential_distribution<float>(1.0/8.0);
    }
    int numUnits = min((int)round(numUnitsDist(rnd)+1), (int)pool.size());
    // if (!guaranteeFood) numUnits = numUnits;
    vector<pair<UNIT_TYPEID, int>> result;
    bool hasSCV = false;
    bool hasCMD = false;
    bool hasGas = false;
    for (int i = 0; i < numUnits; i++) {
        uniform_int_distribution<int> typeDist(0, pool.size()-1);
        UNIT_TYPEID type = pool[typeDist(rnd)];
        exponential_distribution<double> countDist;

        if (!isStructure(type) && !isBasicHarvester(type)) {
            if (bernoulli_distribution(0.8)(rnd)) continue;
        }

        if (isStructure(type)) {
            countDist = exponential_distribution<double>(1.0/3.0);
        } else if (isBasicHarvester(type)) {
            countDist = exponential_distribution<double>(1.0/40.0);
        } else {
            if (bernoulli_distribution(0.05)(rnd)) {
                countDist = exponential_distribution<double>(1.0/20.0);
            } else {
                countDist = exponential_distribution<double>(1.0/2.0);
            }
        }

        int cnt = (int)round(countDist(rnd));

        // Avoid states with very few harvesters as that is not likely in actual games
        if (isBasicHarvester(type) && bernoulli_distribution(0.9)(rnd)) {
            cnt = cnt + 10;
        }

        if (cnt > 0) {
            result.push_back(make_pair(type, min(100, cnt)));

            hasCMD |= isTownHall(type);
            hasSCV |= isBasicHarvester(type);
            hasGas |= isVespeneHarvester(type);
        }
    }

    if (guaranteeFood) {
        exponential_distribution<float> numSCVDist(1.0/40.0);
        if (!hasSCV) result.push_back({ getHarvesterUnitForRace(race), 5+(int)round(numSCVDist(rnd))});
        exponential_distribution<float> numCMDDist(1.0/1.0);
        if (!hasCMD) result.push_back({ getTownHallForRace(race), 1 + (int)round(numCMDDist(rnd))});
        if (!hasGas && bernoulli_distribution(0.5)(rnd)) result.push_back({ getVespeneHarvesterForRace(race), 1 + (int)round(exponential_distribution<double>(1.0/1.0)(rnd)) });
    }

    if (guaranteeFood) {
        int food = 0;
        for (int i = 0; i < result.size(); i++) {
            auto& data = getUnitData(result[i].first);
            food += (data.food_provided - data.food_required) * result[i].second;
        }

        // Add some noise
        food = (int)round(normal_distribution<float>(food, 8)(rnd));

        auto supplyUnit = getSupplyUnitForRace(race);
        int numSupply = (-food + getUnitData(supplyUnit).food_provided - 1) / getUnitData(supplyUnit).food_provided;
        numSupply = max(numSupply, 0);

        for (auto& p : result) {
            if (p.first == supplyUnit) {
                p = make_pair(p.first, p.second + numSupply);
                numSupply = 0;
            }
        }

        if (numSupply > 0) result.push_back({ supplyUnit, numSupply });
    }
    return result;
}

struct UnitCount {
    UNIT_TYPEID type;
    int count;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(type), CEREAL_NVP(count));
    }
};

struct BuildOrderInstance {
    vector<UnitCount> startingUnits;
    vector<UnitCount> targetUnits;
    vector<UNIT_TYPEID> buildOrder;
    float buildOrderTime;
    float startingMinerals;
    float startingVespene;
    Race race;
    int version;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(version),
            CEREAL_NVP(startingUnits),
            CEREAL_NVP(targetUnits),
            CEREAL_NVP(buildOrder),
            CEREAL_NVP(buildOrderTime),
            CEREAL_NVP(startingMinerals),
            CEREAL_NVP(startingVespene)
        );
    }
};

struct Session {
    vector<BuildOrderInstance> instances;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(instances));
    }
};

int main() {
    initMappings();

    // for (auto u : unitTypesTerran2) {
    //     cout << (int)u << ": " << getUnitData(u).food_provided - getUnitData(u).food_required << "," << "  # " << UnitTypeToName(u) << endl;
    // }
    // exit(0);

    // BuildOptimizer optimizer;
    // optimizer.init();
    // unitTestBuildOptimizer(optimizer);
    // unitTestBuildOptimizer();

    default_random_engine rnd(time(0));
    srand(time(0));

    /*if (false) {
        // vector<pair<UNIT_TYPEID,int>> startingUnits = {{(UNIT_TYPEID)48, 6}, {(UNIT_TYPEID)50, 14}, {(UNIT_TYPEID)28, 1}, {(UNIT_TYPEID)30, 2}, {(UNIT_TYPEID)56, 37}, {(UNIT_TYPEID)28, 1}, {(UNIT_TYPEID)24, 2}, {(UNIT_TYPEID)35, 7}, {(UNIT_TYPEID)484, 8}, {(UNIT_TYPEID)20, 5}, {(UNIT_TYPEID)23, 1}, {(UNIT_TYPEID)19, 3}, {(UNIT_TYPEID)53, 19}, {(UNIT_TYPEID)33, 25}, {(UNIT_TYPEID)689, 22}, {(UNIT_TYPEID)49, 1}, {(UNIT_TYPEID)29, 3}, {(UNIT_TYPEID)57, 24}, {(UNIT_TYPEID)19, 1}, {(UNIT_TYPEID)53, 30}, {(UNIT_TYPEID)49, 2}, {(UNIT_TYPEID)30, 2}, {(UNIT_TYPEID)57, 1}, {(UNIT_TYPEID)51, 9}, {(UNIT_TYPEID)21, 3}, {(UNIT_TYPEID)48, 16}, {(UNIT_TYPEID)45, 12}, {(UNIT_TYPEID)18, 1}};
        vector<pair<UNIT_TYPEID,int>> startingUnits = {{UNIT_TYPEID::TERRAN_COMMANDCENTER, 1}, {UNIT_TYPEID::TERRAN_SCV, 12}, {UNIT_TYPEID::TERRAN_BARRACKS, 0}, {UNIT_TYPEID::TERRAN_SUPPLYDEPOT, 3}};
        for (auto p : startingUnits) {
            cout << UnitTypeToName(p.first) << " " << p.second << endl;
        }
        // vector<pair<UNIT_TYPEID,int>> targetUnits = {{(UNIT_TYPEID)49, 2}, {(UNIT_TYPEID)25, 10}, {(UNIT_TYPEID)498, 7}, {(UNIT_TYPEID)30, 1}, {(UNIT_TYPEID)45, 12}, {(UNIT_TYPEID)18, 1}};
        vector<pair<UNIT_TYPEID,int>> targetUnits = {{UNIT_TYPEID::TERRAN_MARAUDER, 0}, {UNIT_TYPEID::TERRAN_BARRACKSREACTOR, 2}};
        // 'buildOrder': [19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 45, 45, 45, 45, 45, 19, 45, 45, 45, 45, 27, 498, 19, 22, 25, 25, 25, 25, 498, 25, 498, 498, 498, 25, 25, 498, 498, 25, 25, 25]
        BuildState startState(startingUnits);
        // 'buildOrderTime': 1501.7911376953125
        startState.resources.minerals = 904.50830841064453;
        startState.resources.vespene = 5000;
        startState.race = Race::Terran;
        for (int k = 0; k < 1; k++) {
            // auto buildOrder = findBestBuildOrderGenetic(startState, targetUnits);
            auto buildOrder = {
                UNIT_TYPEID::TERRAN_BARRACKS,
                UNIT_TYPEID::TERRAN_SCV,
                UNIT_TYPEID::TERRAN_SCV,
                UNIT_TYPEID::TERRAN_REFINERY,
                UNIT_TYPEID::TERRAN_SCV,
                UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
                UNIT_TYPEID::TERRAN_SUPPLYDEPOT,
                UNIT_TYPEID::TERRAN_BARRACKS,
                UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
                UNIT_TYPEID::TERRAN_MARAUDER,
                UNIT_TYPEID::TERRAN_MARAUDER,
                UNIT_TYPEID::TERRAN_MARAUDER,
                UNIT_TYPEID::TERRAN_SCV,
            };

            printBuildOrderDetailed(startState, buildOrder);

            auto state2 = startState;
            state2.simulateBuildOrder(buildOrder);
            cout << "Found time " << state2.time << endl;
        }
        exit(0);
    }

    if (false) {
        pybind11::scoped_interpreter guard{};
        BuildOptimizerNN optimizerNN;

        vector<pair<UNIT_TYPEID,int>> startingUnits = {
            { UNIT_TYPEID::TERRAN_COMMANDCENTER, 2},
            { UNIT_TYPEID::TERRAN_SCV, 20 },
            { UNIT_TYPEID::TERRAN_REFINERY, 2 },
            { UNIT_TYPEID::TERRAN_SUPPLYDEPOT, 1 },
            { UNIT_TYPEID::TERRAN_FACTORY, 4 },
            { UNIT_TYPEID::TERRAN_STARPORT, 4 },
            { UNIT_TYPEID::TERRAN_BARRACKS, 1 },
        };
        vector<pair<int,int>> startingUnitsList;
        for (auto u : startingUnits) startingUnitsList.push_back(make_pair((int)u.first, u.second));

        BuildState startState(startingUnits);
        startState.resources.minerals = 0;
        startState.resources.vespene = 0;
        startState.race = Race::Terran;

        vector<vector<pair<int,int>>> targetUnitsList(25);
        vector<float> actualTimes(targetUnitsList.size());
        for (int j = 0; j < 25; j++) {
            vector<pair<UNIT_TYPEID,int>> targetUnits = {
                { UNIT_TYPEID::TERRAN_HELLION, 4 },
                { UNIT_TYPEID::TERRAN_MARINE, 34 },
                { UNIT_TYPEID::TERRAN_MEDIVAC, 1 },
                { UNIT_TYPEID::TERRAN_REAPER, 1 },
                { UNIT_TYPEID::TERRAN_SCV, 23 },
                // { UNIT_TYPEID::TERRAN_MEDIVAC, 1 },
                // { UNIT_TYPEID::TERRAN_SCV, 15 },
                // { UNIT_TYPEID::TERRAN_HELLION, 1 },
                // { UNIT_TYPEID::TERRAN_CYCLONE, 1 },
                // { UNIT_TYPEID::TERRAN_VIKINGASSAULT, 1 },
            };
            for (auto u : targetUnits) targetUnitsList[j].push_back(make_pair((int)u.first, u.second));
            targetUnits[4] = make_pair(UNIT_TYPEID::TERRAN_SCV, 23+20);
            auto buildOrder = findBestBuildOrderGenetic(startState, targetUnits);
            auto state2 = startState;
            state2.simulateBuildOrder(buildOrder);
            actualTimes[j] = state2.time;
            if (j == 24) printBuildOrderDetailed(startState, buildOrder);
        }

        auto times = optimizerNN.predictTimeToBuild(startingUnitsList, startState.resources, targetUnitsList);
        for (int i = 0; i < times.size(); i++) {
            cout << i << " " << times[i] << " " << actualTimes[i] << endl;
        }

        exit(0);
    }*/

    Session session;
    for (int i = 0; i < 1000000; i++) {
        cout << endl << "Session " << i << endl << endl;
        exponential_distribution<double> mineralDist(1.0/400.0);
        exponential_distribution<double> vespeneDist(1.0/200.0);

        BuildOrderInstance inst;
        inst.race = Race::Protoss;
        auto startingUnits = sampleUnitConfig(rnd, true);
        auto targetUnits = sampleTargetUnitConfig(rnd);
        auto targetUnitState = targetUnits;
        for (auto p : targetUnits) {
            inst.targetUnits.push_back({ p.first, p.second });
        }
        for (auto p : startingUnits) {
            inst.startingUnits.push_back({ p.first, p.second });
            // Starting units are added on top of the target units.
            // The target units are always "what units do we want *in addition* to the ones we already have"
            targetUnitState.push_back({ p.first, p.second });
        }
        inst.startingMinerals = mineralDist(rnd);
        inst.startingVespene = vespeneDist(rnd);

        BuildState startState(startingUnits);
        startState.resources.minerals = inst.startingMinerals;
        startState.resources.minerals = inst.startingVespene;
        startState.race = inst.race;
        for (auto u : startingUnits) {
            if (u.first == UNIT_TYPEID::PROTOSS_WARPGATE && u.second > 0) startState.hasWarpgateResearch = true;
        }

        inst.buildOrderTime = 1000000000;
        inst.version = 6;
        cout << endl << endl;
        for (int k = 0; k < 3; k++) {
            auto buildOrder = findBestBuildOrderGenetic(startState, targetUnitState);
            auto state2 = startState;
            state2.simulateBuildOrder(buildOrder);
            cout << state2.time << endl;
            if (state2.time < inst.buildOrderTime) {
                inst.buildOrder.clear();
                for (auto item : buildOrder.items) {
                    if (item.isUnitType()) inst.buildOrder.push_back(item.typeID());
                }
                inst.buildOrderTime = state2.time;
            }
        }

        // for (auto u : inst.startingUnits) cout << "Has " << getUnitData(u.type).name << " " << u.count << endl;
        // for (auto u : inst.targetUnits) cout << "Wants " << getUnitData(u.type).name << " " << u.count << endl;
        // for (auto u : inst.buildOrder) cout << "Order " << getUnitData(u).name << endl;

        // vector<pair<int,int>> startingUnitsList;
        // vector<vector<pair<int,int>>> targetUnitsList(25);
        // for (auto u : startingUnits) startingUnitsList.push_back(make_pair((int)u.first, u.second));
        // for (int k = 0; k < 25; k++) {
        //     for (auto u : targetUnits) targetUnitsList[k].push_back(make_pair((int)u.first, u.second));
        // }

        // Stopwatch w;
        // auto times = optimizerNN.predictTimeToBuild(startingUnitsList, startState.resources, targetUnitsList);
        // w.stop();
        // cout << "Elapsed " << w.millis() << endl;

        // float nnTime = times[0];

        // cout << "NNTime " << nnTime << " actual time " << inst.buildOrderTime << endl;

        session.instances.push_back(inst);

        if (session.instances.size() > 20) {
            stringstream ss;
            ss << "training_data/buildorders_time/3/chunk_" << rand() << ".json";
            ofstream json(ss.str());
            {
                cereal::JSONOutputArchive archive(json);
                session.serialize(archive);
            }

            session.instances.clear();
        }
    }
    return 0;
}
