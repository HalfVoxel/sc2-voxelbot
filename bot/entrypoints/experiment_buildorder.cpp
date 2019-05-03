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

using namespace sc2;
using namespace std;

vector<pair<UNIT_TYPEID, int>> sampleUnitConfig (default_random_engine& rnd, Race race, bool guaranteeFood) {
    // auto& pool = guaranteeFood ? unitTypesTerran2 : unitTypesTerran3;
    auto pool = getAvailableUnitsForRace(race).getUnitTypes();
    if (!guaranteeFood) {
        for (int i = pool.size() - 1; i >= 0; i--) {
            if (!isArmy(pool[i])) pool.erase(pool.begin() + i);
        }
    }

    for (int i = pool.size() - 1; i >= 0; i--) {
        if (pool[i] == UNIT_TYPEID::PROTOSS_MOTHERSHIP || pool[i] == UNIT_TYPEID::PROTOSS_ARCHON) {
            pool.erase(pool.begin() + i);
        }
    }

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

        if (!isStructure(type) && !isBasicHarvester(type)) {
            if (bernoulli_distribution(0.8)(rnd)) continue;
        }

        if (isStructure(type)) {
            countDist = exponential_distribution<double>(1.0/2.0);
        } else if (isBasicHarvester(type)) {
            countDist = exponential_distribution<double>(1.0/40.0);
        } else {
            countDist = exponential_distribution<double>(1.0/2.0);
        }

        int cnt = (int)round(countDist(rnd));
        if (cnt > 0) {
            result.push_back(make_pair(type, min(100, cnt)));

            hasCMD |= isTownHall(type);
            hasSCV |= isBasicHarvester(type);
        }
    }

    if (guaranteeFood) {
        exponential_distribution<float> numSCVDist(1.0/15.0);
        if (!hasSCV) result.push_back({ getHarvesterUnitForRace(race), 1+(int)round(numSCVDist(rnd))});
        exponential_distribution<float> numCMDDist(1.0/1.0);
        if (!hasCMD) result.push_back({ getTownHallForRace(race), 1 + (int)round(numCMDDist(rnd))});
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

    vector<int> steps = { 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
    vector<int> poolSizes = { 2, 4, 8, 16, 32, 64, 128 };
    vector<float> varianceBiases = { 0, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0 };
    vector<float> mutationRates = { 0, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5 };
    vector<float> mutationRates2 = { 0.015, 0.025, 0.04, 0.05, 0.07, 0.09, 0.1 };

    int num = 100;
    vector<vector<float>> allScores(num);
    vector<vector<float>> allTimes(num);

    vector<vector<float>> allScoresPool(num);
    vector<vector<float>> allTimesPool(num);

    vector<vector<float>> allScoresVar(num);
    vector<vector<float>> allTimesVar(num);

    for (int i = 0; i < num; i++) {
        exponential_distribution<double> mineralDist(1.0/400.0);
        exponential_distribution<double> vespeneDist(1.0/200.0);

        Race race = Race::Protoss;
        auto startingUnits = sampleUnitConfig(rnd, race, true);
        auto targetUnits = sampleUnitConfig(rnd, race, false);
        // for (auto p : targetUnits)  inst.targetUnits.push_back({ p.first, p.second });
        // for (auto p : startingUnits) inst.startingUnits.push_back({ p.first, p.second });

        BuildState startState(startingUnits);
        startState.resources.minerals = mineralDist(rnd);
        startState.resources.minerals = vespeneDist(rnd);
        startState.race = race;
        
        vector<float>& scores = allScores[i];
        vector<float>& times = allTimes[i];

        /*for (auto its : steps) {
            cout << "Sim " << its << endl;
            BuildOptimizerParams params;
            params.iterations = its;

            BuildOrderFitness fitness;
            vector<UNIT_TYPEID> buildOrder;
            tie(buildOrder, fitness) = findBestBuildOrderGenetic(startState, targetUnits, nullptr, params);

            auto state2 = startState;
            state2.simulateBuildOrder(buildOrder, nullptr, true);
            
            scores.push_back(fitness.score());
            times.push_back(state2.time);
        }

        vector<float>& scoresPool = allScoresPool[i];
        vector<float>& timesPool = allTimesPool[i];

        for (auto poolSize : poolSizes) {
            cout << "Pool " << poolSize << endl;
            BuildOptimizerParams params;
            params.iterations = 8192 / poolSize;
            params.genePoolSize = poolSize;

            BuildOrderFitness fitness;
            vector<UNIT_TYPEID> buildOrder;
            tie(buildOrder, fitness) = findBestBuildOrderGenetic(startState, targetUnits, nullptr, params);

            auto state2 = startState;
            state2.simulateBuildOrder(buildOrder, nullptr, true);
            
            scoresPool.push_back(fitness.score());
            timesPool.push_back(state2.time);
        }*/

        vector<float>& scoresVar = allScoresVar[i];
        vector<float>& timesVar = allTimesVar[i];

        vector<pair<BuildOrderItem, int>> targetUnits2;
        for (auto p : targetUnits) targetUnits2.push_back({ BuildOrderItem(p.first), p.second });

        for (auto varianceBias : mutationRates2) {
            cout << "Var " << varianceBias << endl;
            BuildOptimizerParams params;
            params.iterations = 100;
            params.mutationRateAddRemove = varianceBias;

            BuildOrderFitness fitness;
            BuildOrder buildOrder;
            tie(buildOrder, fitness) = findBestBuildOrderGenetic(startState, targetUnits2, nullptr, params);

            auto state2 = startState;
            state2.simulateBuildOrder(buildOrder, nullptr, true);
            
            scoresVar.push_back(fitness.score());
            timesVar.push_back(state2.time);
        }
    }

    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        print(sys.path)
        sys.path.append("bot/python")
    )");

    // pybind11::module::import("experiment_buildorder").attr("save")("iterations", allTimes, allScores, steps);
    // pybind11::module::import("experiment_buildorder").attr("save")("genePoolSize", allTimesPool, allScoresPool, poolSizes);
    pybind11::module::import("experiment_buildorder").attr("save")("varianceBias", allTimesVar, allScoresVar, mutationRates2);
}
