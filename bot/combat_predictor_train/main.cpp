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
        } else if (type == UNIT_TYPEID::TERRAN_SCV) {
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
        CEREAL_NVP(unit.owner),
        CEREAL_NVP(unit.type),
        CEREAL_NVP(unit.health),
        CEREAL_NVP(unit.health_max),
        CEREAL_NVP(unit.shield),
        CEREAL_NVP(unit.shield_max),
        CEREAL_NVP(unit.energy),
        CEREAL_NVP(unit.is_flying)
    );
}

template <class Archive>
void serialize(Archive& archive, CombatResult& result) {
    archive(result.time, result.state);
}

template <class Archive>
void serialize(Archive& archive, CombatState& state) {
    archive(state.units);
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
        archive(instances);
    }
};

int main() {
    CombatPredictor predictor;
    initMappings();
    predictor.init();

    default_random_engine rnd(time(0));
    srand(time(0));

    Session session;
    for (int i = 0; i < 1000000; i++) {
        cout << endl << "Session " << i << endl << endl;

        CombatState state;

        auto defenders = sampleUnitConfig(rnd);
        auto attackers = sampleUnitConfig(rnd);
        for (auto u : defenders) for (int c = 0; c < u.second; c++) state.units.push_back(makeUnit(1, u.first));
        for (auto u : attackers) for (int c = 0; c < u.second; c++) state.units.push_back(makeUnit(2, u.first));

        CombatResult result = predictor.predict_engage(state);

        CombatInstance inst;
        inst.startingState = state;
        inst.outcome = result;

        session.instances.push_back(inst);

        if (session.instances.size() > 1000) {
            stringstream ss;
            cout << "Writing" << endl;
            ss << "training_data/combatsimulations/1/chunk_" << rand() << ".json";
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