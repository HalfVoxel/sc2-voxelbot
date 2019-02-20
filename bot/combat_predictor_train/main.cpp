#include "../utilities/mappings.h"
#include "../utilities/predicates.h"
#include "../BuildOptimizerGenetic.h"
#include "../CombatPredictor.cpp"
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <pybind11/embed.h>
#include <pybind11/stl.h>

using namespace sc2;
using namespace std;

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

vector<pair<UNIT_TYPEID, int>> sampleUnitConfig (default_random_engine& rnd) {

    exponential_distribution<float> numUnitsDist(1.0/3.0);
    int numUnits = min((int)round(numUnitsDist(rnd)+1), (int)unitTypesTerran3.size());
    vector<pair<UNIT_TYPEID, int>> result;
    for (int i = 0; i < numUnits; i++) {
        uniform_int_distribution<int> typeDist(0, unitTypesTerran3.size()-1);
        UNIT_TYPEID type = unitTypesTerran3[typeDist(rnd)];
        exponential_distribution<double> countDist;
        if (isStructure(type)) {
            countDist = exponential_distribution<double>(1.0/3.0);
        } else if (type == UNIT_TYPEID::TERRAN_SCV && false) {
            countDist = exponential_distribution<double>(1.0/40.0);
        } else {
            countDist = exponential_distribution<double>(1.0/10.0);
        }

        int cnt = (int)round(countDist(rnd));
        if (cnt > 0) {
            result.push_back(make_pair(type, min(200, cnt)));
        }
    }

    return result;
}

template <class Archive>
void serialize(Archive& archive, CombatUnit& unit) {
    archive(
        cereal::make_nvp("owner", unit.owner),
        cereal::make_nvp("type", unit.type),
        cereal::make_nvp("health", unit.health),
        cereal::make_nvp("health_max", unit.health_max),
        cereal::make_nvp("shield", unit.shield),
        cereal::make_nvp("shield_max", unit.shield_max),
        cereal::make_nvp("energy", unit.energy),
        cereal::make_nvp("is_flying", unit.is_flying)
    );
}

template <class Archive>
void serialize(Archive& archive, CombatResult& result) {
    archive(cereal::make_nvp("time", result.time), cereal::make_nvp("state", result.state));
}

template <class Archive>
void serialize(Archive& archive, CombatState& state) {
    archive(cereal::make_nvp("units", state.units));
}

struct CombatInstance {
    CombatState startingState;
    CombatResult outcome;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(startingState), CEREAL_NVP(outcome));
    }
};

struct Session {
    vector<CombatInstance> instances;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(instances));
    }
};

pybind11::object saveFunction;

int main() {
    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        sys.path.append("bot/python")
    )");
    pybind11::module mod = pybind11::module::import("replay_saver");
    
    saveFunction = mod.attr("save");


    CombatPredictor predictor;
    initMappings();
    predictor.init();

    default_random_engine rnd(time(0));
    srand(time(0));

    Session session;
    for (int i = 0; i < 1000000; i++) {
        cout << "\rSession " << i;

        CombatState state;

        auto defenders = sampleUnitConfig(rnd);
        auto attackers = sampleUnitConfig(rnd);
        for (auto u : defenders) for (int c = 0; c < u.second; c++) state.units.push_back(makeUnit(1, u.first));
        for (auto u : attackers) for (int c = 0; c < u.second; c++) state.units.push_back(makeUnit(2, u.first));

        // cout << "Original" << endl;
        // for (auto u : state.units) {
        //     cout << u.owner << " " << UnitTypeToName(u.type) << endl;
        // }

        CombatResult result;

        for (int it = 0;; it++) {
            result = predictor.predict_engage(state);
            if (it == 5) break;

            int winner = result.state.owner_with_best_outcome();
            int unitCount = 0;
            for (int j = 0; j < state.units.size(); j++) {
                unitCount += state.units[j].owner == winner;
            }

            if (unitCount > 3) {
                for (int k = 0; k < 3; k++) {
                    int offset = rand() % state.units.size();
                    for (int j = 0; j < state.units.size(); j++) {
                        int idx = (j + offset) % state.units.size();
                        if (state.units[idx].owner == winner) {
                            state.units.erase(state.units.begin() + idx);
                            break;
                        }
                    }
                }
            }
        }

        // cout << "Balanced" << endl;
        CombatInstance inst;
        inst.startingState = state;
        inst.outcome = result;

        // for (auto u : state.units) {
        //     cout << u.owner << " " << UnitTypeToName(u.type) << endl;
        // }

        // cout << endl << "Outcome" << endl;
        // for (auto u : result.state.units) {
        //     cout << u.owner << " " << UnitTypeToName(u.type) << " " << u.health << endl;
        // }
        // exit(0);

        session.instances.push_back(inst);

        if (session.instances.size() > 1000) {
            cout << endl;
            stringstream ss;
            cout << "Writing" << endl;
            ss << "training_data/combatsimulations/1/chunk_" << rand() << ".json";
            stringstream json;
            {
                cereal::JSONOutputArchive archive(json);
                session.serialize(archive);
            }

            saveFunction(json.str(), ss.str());

            session.instances.clear();
        }
    }
    return 0;
}