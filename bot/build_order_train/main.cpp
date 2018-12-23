#include "../BuildOptimizerGenetic.h"
#include "../utilities/mappings.h"
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
#include <vector>
#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_interfaces.h"

using namespace std;
using namespace sc2;

struct SerializedUnitInProgress {
    UNIT_TYPEID type;
    float remainingTime; // seconds

    template <class Archive>
    void serialize(Archive& archive) {
        archive(type, remainingTime);
    }
};

struct SerializedUnit {
    UNIT_TYPEID type;
    UNIT_TYPEID addon;
    int totalCount;
    int availableCount;

    SerializedUnit(const BuildUnitInfo& unit)
        : type(unit.type), addon(unit.addon), totalCount(unit.units), availableCount(unit.availableUnits()) {
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(type, addon, totalCount, availableCount);
    }
};

struct SerializedState {
    vector<SerializedUnit> units;
    vector<SerializedUnitInProgress> unitsInProgress;
    float minerals;
    float vespene;
    float mineralsPerSecond;
    float vespenePerSecond;
    float foodAvailable;
    Race race;
    int highYieldMineralSlots;
    int lowYieldMineralSlots;
    int version;

    SerializedState(const BuildState& state) {
        for (auto& u : state.units) {
            units.push_back(SerializedUnit(u));
        }

        for (auto& ev : state.events) {
            if (ev.type == BuildEventType::FinishedUnit) {
                auto createdUnit = abilityToUnit(ev.ability);
                if (createdUnit != UNIT_TYPEID::INVALID) {
                    auto remainingTime = ev.time - state.time;
                    assert(remainingTime >= 0);
                    unitsInProgress.push_back({
                        createdUnit,
                        remainingTime,
                    });
                }
            }
        }

        minerals = state.resources.minerals;
        vespene = state.resources.vespene;
        auto miningSpeed = state.miningSpeed();
        mineralsPerSecond = miningSpeed.mineralsPerSecond;
        vespenePerSecond = miningSpeed.vespenePerSecond;
        foodAvailable = state.foodAvailable();
        race = state.race;
        version = 1;

        highYieldMineralSlots = 0;
        lowYieldMineralSlots = 0;
        for (auto b : state.baseInfos) {
            auto slots = b.mineralSlots();
            highYieldMineralSlots += slots.first;
            lowYieldMineralSlots += slots.second;
        }
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(units, unitsInProgress, minerals, vespene, mineralsPerSecond, vespenePerSecond);
    }
};

struct Session {
    vector<SerializedState> states;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(states);
    }
};

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

vector<pair<UNIT_TYPEID, int>> sampleUnitConfig (default_random_engine& rnd, bool guaranteeFood) {
    auto& pool = guaranteeFood ? unitTypesTerran2 : unitTypesTerran3;

    exponential_distribution<float> numUnitsDist;
    if (bernoulli_distribution(0.05)(rnd)) {
        numUnitsDist = exponential_distribution<float>(1.0/10.0);
    } else {
        numUnitsDist = exponential_distribution<float>(1.0/8.0);
    }
    int numUnits = min((int)round(numUnitsDist(rnd)+1), (int)pool.size());
    // if (!guaranteeFood) numUnits = numUnits;
    vector<pair<UNIT_TYPEID, int>> result;
    bool hasSCV = false;
    bool hasCMD = false;
    for (int i = 0; i < numUnits; i++) {
        uniform_int_distribution<int> typeDist(0, pool.size()-1);
        UNIT_TYPEID type = pool[typeDist(rnd)];
        exponential_distribution<double> countDist;

        if (!isStructure(type) && type != UNIT_TYPEID::TERRAN_SCV) {
            if (bernoulli_distribution(0.8)(rnd)) continue;
        }

        if (isStructure(type)) {
            countDist = exponential_distribution<double>(1.0/3.0);
        } else if (type == UNIT_TYPEID::TERRAN_SCV) {
            countDist = exponential_distribution<double>(1.0/40.0);
        } else {
            if (bernoulli_distribution(0.05)(rnd)) {
                countDist = exponential_distribution<double>(1.0/20.0);
            } else {
                countDist = exponential_distribution<double>(1.0/2.0);
            }
        }

        int cnt = (int)round(countDist(rnd));
        if (cnt > 0) {
            result.push_back(make_pair(type, min(100, cnt)));

            hasCMD |= isTownHall(type);
            hasSCV |= type == UNIT_TYPEID::TERRAN_SCV;
        }
    }

    if (guaranteeFood) {
        exponential_distribution<float> numSCVDist(1.0/15.0);
        if (!hasSCV) result.push_back({UNIT_TYPEID::TERRAN_SCV, 1+(int)round(numSCVDist(rnd))});
        exponential_distribution<float> numCMDDist(1.0/1.0);
        if (!hasCMD) result.push_back({UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 + (int)round(numCMDDist(rnd))});
    }

    if (guaranteeFood) {
        int food = 0;
        for (int i = 0; i < result.size(); i++) {
            auto& data = getUnitData(result[i].first);
            food += (data.food_provided - data.food_required) * result[i].second;
        }

        // Add some noise
        food = (int)round(normal_distribution<float>(food, 8)(rnd));

        auto supplyUnit = getSupplyUnitForRace(Race::Terran);
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

int main() {
    initMappings();
    // BuildOptimizer optimizer;
    // optimizer.init();
    // unitTestBuildOptimizer(optimizer);

    default_random_engine rnd;

    while(true) {
        auto guaranteeFood = bernoulli_distribution(0.95);
        auto gameStartConfig = bernoulli_distribution(0.1);
        auto startUnits = sampleUnitConfig(rnd, guaranteeFood(rnd));
        if (gameStartConfig) startUnits = { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } };
        auto targetUnits = sampleUnitConfig(rnd, false);

        exponential_distribution<double> mineralDist(1.0/400.0);
        exponential_distribution<double> vespeneDist(1.0/200.0);

        BuildState startState(startUnits);
        startState.resources.minerals = mineralDist(rnd);
        startState.resources.minerals = vespeneDist(rnd);
        startState.race = Race::Terran;

        if (gameStartConfig) {
            // Initial delay before harvesters start mining properly
            startState.makeUnitsBusy(UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::INVALID, 12);
            for (int i = 0; i < 12; i++)
                startState.addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, 4, UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::INVALID));
        }
        

        stringstream ss;
        ss << "training_data/buildorders/1/chunk_" << rand() << ".json";
        ofstream json(ss.str());
        {
            cereal::JSONOutputArchive archive(json);
            session.serialize(archive);
        }
    }
    
    return 0;
}
