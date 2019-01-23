#include "../BuildOptimizerGenetic.h"
#include "../utilities/mappings.h"
#include "../utilities/predicates.h"
#include "../utilities/stdutils.h"
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
#include "serialized_state.h"

using namespace std;
using namespace sc2;



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

static vector<UNIT_TYPEID> unitTypesTerran5 = {
    UNIT_TYPEID::TERRAN_ARMORY,
    UNIT_TYPEID::TERRAN_BANSHEE,
    UNIT_TYPEID::TERRAN_BARRACKS,
    UNIT_TYPEID::TERRAN_BARRACKSREACTOR,
    UNIT_TYPEID::TERRAN_BARRACKSTECHLAB,
    UNIT_TYPEID::TERRAN_BATTLECRUISER,
    UNIT_TYPEID::TERRAN_BUNKER,
    UNIT_TYPEID::TERRAN_COMMANDCENTER,
    UNIT_TYPEID::TERRAN_CYCLONE,
    UNIT_TYPEID::TERRAN_ENGINEERINGBAY,
    UNIT_TYPEID::TERRAN_FACTORY,
    UNIT_TYPEID::TERRAN_FACTORYREACTOR,
    UNIT_TYPEID::TERRAN_FACTORYTECHLAB,
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
    UNIT_TYPEID::TERRAN_STARPORTREACTOR,
    UNIT_TYPEID::TERRAN_STARPORTTECHLAB,
    UNIT_TYPEID::TERRAN_SUPPLYDEPOT,
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

vector<pair<UNIT_TYPEID, int>> sampleTargetUnits(default_random_engine& rnd) {
    exponential_distribution<float> numUnitsDist;
    if (bernoulli_distribution(0.05)(rnd)) {
        numUnitsDist = exponential_distribution<float>(1.0/5.0);
    } else {
        numUnitsDist = exponential_distribution<float>(1.0/1.0);
    }

    int numUnits = 1 + numUnitsDist(rnd);

    auto& pool = unitTypesTerran3;
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

default_random_engine randomEngine(time(0));

void toOneHot(const vector<int>& values, int dim, vector<float>::iterator start) {
    for (int i = 0; i < values.size(); i++) {
        *(start + dim*i + max(min(values[i], dim - 1), 0)) = 1;
    }
}

vector<int> unitIndexMapping;
vector<int> economicUnitIndexMapping;
int unitIndexMappingSize;
int economicUnitIndexMappingSize;
void setUnitIndexMapping(map<int, int> unitTypes, map<int,int> economicUnitTypes) {
    auto& allUnitTypes = getUnitTypes();
    unitIndexMapping = vector<int>(allUnitTypes.size(), -1);
    economicUnitIndexMapping = vector<int>(allUnitTypes.size(), -1);

    unitIndexMappingSize = 0;
    economicUnitIndexMappingSize = 0;

    for (auto& p : unitTypes) {
        assert(p.second >= 0 && p.second < unitTypes.size());
        unitIndexMapping[p.first] = p.second;
        unitIndexMappingSize = max(unitIndexMappingSize, p.second + 1);
    }
    for (auto& p : economicUnitTypes) {
        assert(p.second >= 0 && p.second < economicUnitTypes.size());
        economicUnitIndexMapping[p.first] = p.second;
        economicUnitIndexMappingSize = max(economicUnitIndexMappingSize, p.second + 1);
    }
}

struct EnvironmentState {
    BuildState startState;
    BuildState state;
    vector<UNIT_TYPEID> buildOrder;
    vector<int> goal;

    EnvironmentState() {
        state = startState = BuildState({ { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } });
    }

    void reset() {
        auto& rnd = randomEngine;
        auto guaranteeFood = bernoulli_distribution(0.95);
        auto gameStartConfig = true; // bernoulli_distribution(0.1)(rnd);
        auto startUnits = sampleUnitConfig(rnd, guaranteeFood(rnd));
        if (gameStartConfig) startUnits = { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } };
        auto targetUnits = sampleUnitConfig(rnd, false);

        exponential_distribution<double> mineralDist(1.0/400.0);
        exponential_distribution<double> vespeneDist(1.0/200.0);

        startState = BuildState(startUnits);

        bool hasVespene = false;
        for (auto u : startUnits) hasVespene |= u.second > 0 && isVespeneHarvester(u.first);
        startState.resources.minerals = mineralDist(rnd);
        startState.resources.vespene = hasVespene || bernoulli_distribution(0.05)(rnd) ? vespeneDist(rnd) : 0;
        startState.race = Race::Terran;
        startState.baseInfos = { BaseInfo(10800, 1000, 1000) };
        state = startState;
        buildOrder.clear();
    }

    void resetToGameStartConfig () {
        startState = BuildState({ { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } });
        startState.resources.minerals = 50;
        startState.race = Race::Terran;
        // Initial delay before harvesters start mining properly
        // startState.makeUnitsBusy(UNIT_TYPEID::PROTOSS_PROBE, UNIT_TYPEID::INVALID, 12);
        // for (int i = 0; i < 12; i++) startState.addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, 4, UNIT_TYPEID::PROTOSS_PROBE, ABILITY_ID::INVALID));

        startState.baseInfos = { BaseInfo(10800, 1000, 1000) };
        state = startState;
        buildOrder.clear();
    }

    void step(UNIT_TYPEID action) {
        action = canonicalize(action);
        buildOrder.push_back(action);

        vector<UNIT_TYPEID> expandedBuildOrder;
        vector<bool> isPartOfOriginalBuildOrder;
        tie(expandedBuildOrder, isPartOfOriginalBuildOrder) = expandBuildOrderWithImplicitSteps(state, { action });

        state.simulateBuildOrder(expandedBuildOrder, nullptr, false);
    }

    float getTime() {
        return state.time;
    }

    string getState() {
        stringstream json;
        {
            cereal::JSONOutputArchive archive(json);
            SerializedState(state).serialize(archive);
        }
        return json.str();
    }

    string getGoal () {
        return "{}";
    }

    void setGoal(vector<int> goal) {
        if (goal.size() != unitIndexMappingSize) {
            cerr << "Found size " << goal.size() << " expected " << unitIndexMappingSize << endl;
            throw std::exception();
        }

        this->goal = goal;
    }

    void print() {
        vector<UNIT_TYPEID> expandedBuildOrder;
        vector<bool> isPartOfOriginalBuildOrder;
        tie(expandedBuildOrder, isPartOfOriginalBuildOrder) = expandBuildOrderWithImplicitSteps(startState, buildOrder);

        printBuildOrderDetailed(startState, expandedBuildOrder, &isPartOfOriginalBuildOrder);
    }

    vector<float> getObservation() {
        vector<int> unitCounts(economicUnitIndexMappingSize);
        vector<int> availableUnits(economicUnitIndexMappingSize);
        vector<int> inProgressUnits(economicUnitIndexMappingSize);
        vector<int> remainingGoal = goal;

        for (auto& u : state.units) {
            if (u.units > 0) {
                int index = economicUnitIndexMapping[(int)u.type];
                if (index != -1) {
                    unitCounts[index] += u.units;
                    availableUnits[index] += u.availableUnits();
                }

                index = unitIndexMapping[(int)u.type];
                if (index == -1) {
                    throw std::exception();
                }
                remainingGoal[index] -= u.units;
            }
        }

        int pendingFood = 0;

        for (auto& ev : state.events) {
            if (ev.type == BuildEventType::FinishedUnit) {
                auto createdUnit = abilityToUnit(ev.ability);
                if (createdUnit != UNIT_TYPEID::INVALID) {
                    auto remainingTime = ev.time - state.time;
                    assert(remainingTime >= 0);

                    {
                        int index = economicUnitIndexMapping[(int)createdUnit];
                        if (index != -1) {
                            unitCounts[index] += 1;
                            inProgressUnits[index] += 1;
                        }
                    }

                    {
                        int index = unitIndexMapping[(int)createdUnit];
                        if (index == -1) {
                            throw std::exception();
                        }
                        remainingGoal[index] -= 1;
                    }

                    pendingFood = getUnitData(createdUnit).food_provided;
                }
            }
        }

        vector<float> result (economicUnitIndexMappingSize*(10+3+5) + 7 + 8 + 3 + unitIndexMappingSize * 2);
        int offset = 0;
        toOneHot(unitCounts, 10, result.begin() + offset);
        offset += 10 * unitCounts.size();

        toOneHot(availableUnits, 3, result.begin() + offset);
        offset += 3 * availableUnits.size();

        toOneHot(inProgressUnits, 5, result.begin() + offset);
        offset += 5 * inProgressUnits.size();

        result[offset + 0] = state.resources.minerals / 100.0f;
        result[offset + 1] = state.resources.vespene / 100.0f;
        auto miningSpeed = state.miningSpeed();
        result[offset + 2] = miningSpeed.mineralsPerSecond / 10.0f;
        result[offset + 3] = miningSpeed.vespenePerSecond / 10.0f;

        int highYieldMineralSlots = 0;
        int lowYieldMineralSlots = 0;
        for (auto b : state.baseInfos) {
            auto slots = b.mineralSlots();
            highYieldMineralSlots += slots.first;
            lowYieldMineralSlots += slots.second;
        }

        result[offset + 4] = highYieldMineralSlots / 10.0f;
        result[offset + 5] = lowYieldMineralSlots / 10.0f;
        int scvIndex = economicUnitIndexMapping[(int)UNIT_TYPEID::TERRAN_SCV];
        assert(scvIndex != -1);
        result[offset + 6] = unitCounts[scvIndex] / 10.0f;
        offset += 7;

        toOneHot({ (int)state.foodAvailable() }, 8, result.begin() + offset);
        offset += 8;

        result[offset] = pendingFood > 0; offset++;
        result[offset] = pendingFood > 8; offset++;
        result[offset] = pendingFood > 16; offset++;

        for (int i = 0; i < remainingGoal.size(); i++, offset++) {
            result[offset] = remainingGoal[i] > 0;
        }

        for (int i = 0; i < remainingGoal.size(); i++, offset++) {
            result[offset] = max(remainingGoal[i], 0);
        }

        if (offset != result.size()) {
            cerr << "Offset " << offset << " expected " << result.size() << endl;
            exit(1);
        }

        return result;

        // def createState(self, state):
        // unitCounts = np.zeros(NUM_UNITS, dtype=np.float32)
        // unitsAvailable = np.zeros(NUM_UNITS, dtype=np.float32)
        // unitsInProgress = np.zeros(NUM_UNITS, dtype=np.float32)

        // for unit in state["units"]:
        //     # Finished
        //     # TODO: Addon
        //     unitIndex = unitIndexMap[unit["type"]]
        //     unitCounts[unitIndex] += unit["totalCount"]
        //     unitsAvailable[unitIndex] += unit["availableCount"]

        // for unit in state["unitsInProgress"]:
        //     # In progress
        //     unitIndex = unitIndexMap[unit["type"]]
        //     unitCounts[unitIndex] += 1
        //     unitsInProgress[unitIndex] += 1

        // originalUnitCounts = unitCounts

        // oneHotUnitsAvailable = to_one_hot(unitsAvailable[NON_MILITARY_UNITS_MASK_INDICES], 3)
        // oneHotUnitCounts = to_one_hot(unitCounts[NON_MILITARY_UNITS_MASK_INDICES], 10)
        // oneHotUnitsInProgress = to_one_hot(unitsInProgress[NON_MILITARY_UNITS_MASK_INDICES], 5)

        // # Some metadata, the data is normalized to approximately 1
        // metaTensor = np.zeros(7, dtype=np.float32)
        // metaTensor[0] = state["minerals"] / 100
        // metaTensor[1] = state["vespene"] / 100
        // metaTensor[2] = state["mineralsPerSecond"] / 10
        // metaTensor[3] = state["vespenePerSecond"] / 10
        // metaTensor[4] = state["highYieldMineralSlots"] / 10
        // metaTensor[5] = state["lowYieldMineralSlots"] / 10
        // metaTensor[6] = unitCounts[unitIndexMap[45]] / 10  # SCV count
        // foodTensor = to_one_hot(np.array([state["foodAvailable"]]), 8)

        // # stateTensor = np.concatenate([np.array([state["time"]]), oneHotUnitCounts, oneHotUnitsAvailable, oneHotUnitsInProgress, metaTensor])
        // stateTensor = np.concatenate([oneHotUnitCounts, oneHotUnitsAvailable, oneHotUnitsInProgress, metaTensor, foodTensor])
        // return stateTensor, originalUnitCounts
    }
};

int add(int i, int j) {
    return i + j;
}


// PYBIND11_MODULE(example, m) {
//     m.doc() = "pybind11 example plugin"; // optional module docstring

//     m.def("add", &add, "A function which adds two numbers");
// }

PYBIND11_MODULE(cppbot, m) {
    initMappings();

    // vector<UNIT_TYPEID> order = {
    //     UNIT_TYPEID::TERRAN_SUPPLYDEPOT,
    //     UNIT_TYPEID::TERRAN_BARRACKS,
    //     UNIT_TYPEID::TERRAN_BARRACKS,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    //     UNIT_TYPEID::TERRAN_MARINE,
    // };
    // EnvironmentState env;
    // for (auto s : order) {
    //     env.step(s);
    // }
    // cout << "Time: " << env.getTime() << endl;
    // env.print();

    m.def("setUnitIndexMapping", &setUnitIndexMapping, "");

    pybind11::class_<EnvironmentState>(m, "EnvironmentState")
        .def(pybind11::init())
        .def("step", [](EnvironmentState &e, int action) { e.step((UNIT_TYPEID)action); })
        .def("reset", &EnvironmentState::reset)
        .def("resetToGameStartConfig", &EnvironmentState::resetToGameStartConfig)
        .def("getState", &EnvironmentState::getState)
        .def("getTime", &EnvironmentState::getTime)
        .def("getGoal", &EnvironmentState::getGoal)
        .def("setGoal", &EnvironmentState::setGoal)
        .def("getObservation", &EnvironmentState::getObservation)
        .def("print", &EnvironmentState::print);
}

pybind11::object predictFunction;
pybind11::object loadSessionFunction;
pybind11::object rewardFunction;

vector<UNIT_TYPEID> predictBuildOrder (const BuildState& startState, vector<pair<UNIT_TYPEID, int>> goal) {
    vector<UNIT_TYPEID> buildOrder;
    vector<UnitCount> serializedGoal;
    for (auto u : goal) serializedGoal.push_back({ u.first, u.second });

    Session fullSession;
    fullSession.goal = serializedGoal;
    vector<float> stateValues;

    auto state = startState;
    for (int i = 0; i < 50; i++) {
        Session session;
        session.states.emplace_back(state);
        session.goal = serializedGoal;
        fullSession.states.emplace_back(state);

        stringstream json;
        {
            cereal::JSONOutputArchive archive(json);
            session.serialize(archive);
        }
        auto res = predictFunction(json.str(), true, false).cast<pair<int, float>>();
        UNIT_TYPEID unitType = (UNIT_TYPEID)res.first;
        float stateValue = res.second;

        unitType = canonicalize(unitType);
        buildOrder.push_back(unitType);
        stateValues.push_back(stateValue);

        vector<UNIT_TYPEID> expandedBuildOrder2;
        vector<bool> isPartOfOriginalBuildOrder2;
        tie(expandedBuildOrder2, isPartOfOriginalBuildOrder2) = expandBuildOrderWithImplicitSteps(startState, buildOrder);
        state = startState;
        state.simulateBuildOrder(expandedBuildOrder2, nullptr, false);
    }

    vector<UNIT_TYPEID> expandedBuildOrder;
    vector<bool> isPartOfOriginalBuildOrder;
    tie(expandedBuildOrder, isPartOfOriginalBuildOrder) = expandBuildOrderWithImplicitSteps(startState, buildOrder);
    printBuildOrderDetailed(startState, expandedBuildOrder, &isPartOfOriginalBuildOrder);

    stringstream json2;
    {
        cereal::JSONOutputArchive archive(json2);
        fullSession.serialize(archive);
    }
    vector<float> rewards = rewardFunction(json2.str()).cast<vector<float>>();
    for (int i = 0; i < buildOrder.size(); i++) {
        cout << "State value for " << UnitTypeToName(buildOrder[i]) << ": " << stateValues[i] << " reward: " << (i > 0 ? rewards[i-1] : 0) << endl;
    }

    return expandedBuildOrder;
}

void generateTrainingData(default_random_engine& rnd) {
    auto guaranteeFood = bernoulli_distribution(0.95);
    auto gameStartConfig = bernoulli_distribution(0.1)(rnd);
    // auto startUnits = sampleUnitConfig(rnd, guaranteeFood(rnd));
    auto startUnits = sampleUnitConfig(rnd, true);
    if (gameStartConfig) startUnits = { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } };
    auto targetUnits = sampleUnitConfig(rnd, false);

    exponential_distribution<double> mineralDist(1.0/400.0);
    exponential_distribution<double> vespeneDist(1.0/200.0);

    BuildState startState(startUnits);

    bool hasVespene = false;
    for (auto u : startUnits) hasVespene |= u.second > 0 && isVespeneHarvester(u.first);
    startState.resources.minerals = mineralDist(rnd);
    startState.resources.vespene = hasVespene || bernoulli_distribution(0.05)(rnd) ? vespeneDist(rnd) : 0;
    startState.race = Race::Terran;

    if (gameStartConfig) {
        // Initial delay before harvesters start mining properly
        startState.makeUnitsBusy(UNIT_TYPEID::TERRAN_SCV, UNIT_TYPEID::INVALID, 12);
        for (int i = 0; i < 12; i++)
            startState.addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, 4, UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::INVALID));
    }

    Session session;
    for (auto u : targetUnits) {
        session.goal.push_back({ u.first, u.second });
    }

    vector<UNIT_TYPEID> originalBuildOrder;
    
    auto tempState = startState;
    for (int i = 0; i < 60; i++) {
        Session tempSession;
        tempSession.states.emplace_back(tempState);
        tempSession.goal = session.goal;
        stringstream json;
        {
            cereal::JSONOutputArchive archive(json);
            tempSession.serialize(archive);
        }
        auto res = predictFunction(json.str(), false, true).cast<pair<int, float>>();
        UNIT_TYPEID unitType = (UNIT_TYPEID)res.first;
        float stateValue = res.second;

        unitType = canonicalize(unitType);
        originalBuildOrder.push_back(unitType);

        vector<UNIT_TYPEID> expandedBuildOrder2;
        vector<bool> isPartOfOriginalBuildOrder2;
        tie(expandedBuildOrder2, isPartOfOriginalBuildOrder2) = expandBuildOrderWithImplicitSteps(startState, originalBuildOrder);
        tempState = startState;
        tempState.simulateBuildOrder(expandedBuildOrder2, nullptr, false);
    }

    /*for (int i = 0; i < 20; i++) {
        originalBuildOrder.push_back(unitTypesTerran5[uniform_int_distribution<int>(0, unitTypesTerran5.size()-1)(rnd)]);
    }*/

    vector<UNIT_TYPEID> expandedBuildOrder;
    vector<bool> partOfOriginalBuildOrder;
    tie(expandedBuildOrder, partOfOriginalBuildOrder) = expandBuildOrderWithImplicitSteps (startState, originalBuildOrder);

    for (int i = 0, j = 0; i < partOfOriginalBuildOrder.size(); i++) {
        if (partOfOriginalBuildOrder[i]) {
            assert(expandedBuildOrder[i] == originalBuildOrder[j]);
            j++;
        }

        if (i == partOfOriginalBuildOrder.size() - 1) {
            assert(j == originalBuildOrder.size());
        }
    }

    // auto buildOrder = findBestBuildOrderGenetic(startState, targetUnits, nullptr);
    auto state = startState;
    session.states.push_back(SerializedState(state));
    int lastSuccessfullAction = -1;
    bool success = state.simulateBuildOrder(expandedBuildOrder, [&](int index) {
        if (partOfOriginalBuildOrder[index]) {
            lastSuccessfullAction = index;
            session.actions.push_back(expandedBuildOrder[index]);
            session.states.push_back(SerializedState(state));
        }
    });
    if (!success) {
        session.actions.push_back(expandedBuildOrder[lastSuccessfullAction+1]);
        session.states.push_back(SerializedState(state));
        session.failed = true;
    }

    // printBuildOrderDetailed(startState, expandedBuildOrder, &partOfOriginalBuildOrder);

    // cout << "*" << endl;
    stringstream json;
    {
        cereal::JSONOutputArchive archive(json);
        session.serialize(archive);
    }
    loadSessionFunction(json.str());
    stringstream ss;
    ss << "training_data/buildorders/1/chunk_" << uniform_int_distribution<int>(1, 1000000)(rnd) << ".json";
    // ofstream jsonFile(ss.str());
    // jsonFile << json.str();
    // jsonFile.close();
}

void generateBuildOrderTrainingData() {
    default_random_engine rnd(time(0));

    while(true) {
        auto guaranteeFood = bernoulli_distribution(1.0);
        auto gameStartConfig = bernoulli_distribution(0.1)(rnd);
        auto startUnits = sampleUnitConfig(rnd, guaranteeFood(rnd));
        if (gameStartConfig) startUnits = { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } };
        // auto targetUnits = sampleUnitConfig(rnd, false);
        auto targetUnits = sampleTargetUnits(rnd);
        for (auto u : startUnits) {
            targetUnits.push_back(u);
        }

        exponential_distribution<double> mineralDist(1.0/400.0);
        exponential_distribution<double> vespeneDist(1.0/200.0);

        BuildState startState(startUnits);

        bool hasVespene = false;
        for (auto u : startUnits) hasVespene |= u.second > 0 && isVespeneHarvester(u.first);
        startState.resources.minerals = mineralDist(rnd);
        startState.resources.vespene = hasVespene || bernoulli_distribution(0.05)(rnd) ? vespeneDist(rnd) : 0;
        startState.race = Race::Terran;

        if (gameStartConfig) {
            // Initial delay before harvesters start mining properly
            startState.makeUnitsBusy(UNIT_TYPEID::TERRAN_SCV, UNIT_TYPEID::INVALID, 12);
            for (int i = 0; i < 12; i++)
                startState.addEvent(BuildEvent(BuildEventType::MakeUnitAvailable, 4, UNIT_TYPEID::TERRAN_SCV, ABILITY_ID::INVALID));
        }

        Session session;
        for (auto u : targetUnits) {
            session.goal.push_back({ u.first, u.second });
        }

        vector<UNIT_TYPEID> originalBuildOrder;

        if (false) {
            for (int i = 0; i < 20; i++) {
                originalBuildOrder.push_back(unitTypesTerran5[uniform_int_distribution<int>(0, unitTypesTerran5.size()-1)(rnd)]);
            }
        } else {
            originalBuildOrder = findBestBuildOrderGenetic(startState, targetUnits, nullptr);
        }

        vector<UNIT_TYPEID> expandedBuildOrder;
        vector<bool> partOfOriginalBuildOrder;
        tie(expandedBuildOrder, partOfOriginalBuildOrder) = expandBuildOrderWithImplicitSteps (startState, originalBuildOrder);

        for (int i = 0, j = 0; i < partOfOriginalBuildOrder.size(); i++) {
            if (partOfOriginalBuildOrder[i]) {
                assert(expandedBuildOrder[i] == originalBuildOrder[j]);
                j++;
            }

            if (i == partOfOriginalBuildOrder.size() - 1) {
                assert(j == originalBuildOrder.size());
            }
        }

        auto state = startState;
        session.states.push_back(SerializedState(state));
        int lastSuccessfullAction = -1;
        bool success = state.simulateBuildOrder(expandedBuildOrder, [&](int index) {
            if (partOfOriginalBuildOrder[index]) {
                lastSuccessfullAction = index;
                session.actions.push_back(expandedBuildOrder[index]);
                session.states.push_back(SerializedState(state));
            }
        });
        if (!success) {
            session.actions.push_back(expandedBuildOrder[lastSuccessfullAction+1]);
            session.states.push_back(SerializedState(state));
            session.failed = true;
        }

        // printBuildOrderDetailed(startState, expandedBuildOrder, &partOfOriginalBuildOrder);

        stringstream ss;
        ss << "training_data/buildorders/1/chunk_" << uniform_int_distribution<int>(1, 1000000)(rnd) << ".json";
        cout << "*" << endl;
        ofstream json(ss.str());
        {
            cereal::JSONOutputArchive archive(json);
            session.serialize(archive);
        }
        json.close();
    }
}

int main() {
    initMappings();
    generateBuildOrderTrainingData();


    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        sys.path.append("bot/build_order_train/baselines")
    )");
    pybind11::module trainer = pybind11::module::import("testfile");
    
    predictFunction = trainer.attr("predict");
    rewardFunction = trainer.attr("calculate_session_rewards");
    auto loadFunction = trainer.attr("load_all");
    auto saveFunction = trainer.attr("save");
    
    auto optimizeEpochFunction = trainer.attr("optimize_epoch");
    loadSessionFunction = trainer.attr("loadSessionFromJson");
    auto loadWeightsFunction = trainer.attr("load_weights");
    
    if(false) {
        loadWeightsFunction("models/buildorders_qlearn_191.weights");
        while(true) {
            vector<pair<UNIT_TYPEID, int>> startUnits = { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } };
            BuildState startState(startUnits);
            predictBuildOrder(startState, { { UNIT_TYPEID::TERRAN_MARINE, 1 } });
            exit(0);
        }
    }
    // BuildOptimizer optimizer;
    // optimizer.init();
    // unitTestBuildOptimizer(optimizer);

    default_random_engine rnd(time(0));

    loadFunction(0);
    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int i = 0; i < 64; i++) {
            generateTrainingData(rnd);
        }

        optimizeEpochFunction();

        vector<pair<UNIT_TYPEID, int>> startUnits = { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } };
        BuildState startState(startUnits);
        predictBuildOrder(startState, { { UNIT_TYPEID::TERRAN_MARINE, 1 } });

        startUnits = { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 }, { UNIT_TYPEID::TERRAN_BARRACKS, 2 } };
        startState = BuildState(startUnits);
        predictBuildOrder(startState, { { UNIT_TYPEID::TERRAN_MARINE, 1 } });

        // vector<pair<UNIT_TYPEID, int>> startUnits = { { UNIT_TYPEID::TERRAN_COMMANDCENTER, 1 }, { UNIT_TYPEID::TERRAN_SCV, 12 } };
        // BuildState startState(startUnits);
        // predictBuildOrder(startState, { { UNIT_TYPEID::TERRAN_MARINE, 1 }});
        
        saveFunction(epoch);
    }
    
    return 0;
}
