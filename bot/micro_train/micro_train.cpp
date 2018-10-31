#include "../bot/main.h"
#include <fstream>
#include <iostream>
#include <queue>
#include "../DependencyAnalyzer.h"
#include "../Mappings.h"
#include "../generated/abilities.h"
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"

#include "cereal/cereal.hpp"
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <pybind11/embed.h> 

const char* kReplayFolder = "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays";
const char* kReplayListProtoss = "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays_protoss.txt";

using Clock=std::chrono::high_resolution_clock;

namespace py = pybind11;
using namespace sc2;
using namespace std;

static const char* EmptyMap = "Test/Empty.SC2Map";

// auto t0 = Clock::now();
//         {
//             ofstream os("data.xml");
//             cereal::JSONOutputArchive archive(os);
//             archive(string("hello"), CEREAL_NVP(string("blah")), cereal::make_nvp("whatevs", 423), MyClass { "asdf", 1, 2, 3 });
//             vector<int> blah = { 0, 1, 2, 3, 4, 5 };
//             archive.saveBinaryValue((int*)&blah[0], sizeof(int) * blah.size(), "thisisbinary");
//         }
//         auto t2 = Clock::now();
//         cout << "Time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t0).count() << " ns" << endl;


struct SerializedPos {
    float x;
    float y;

    template<class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(x), CEREAL_NVP(y));
    }
};

enum Action {
    N,
    NW,
    W,
    SW,
    S,
    SE,
    E,
    NE,
    Attack,
};

struct SerializedUnit {
    SerializedPos position;
    UNIT_TYPEID unit_type;
    UNIT_TYPEID canonical_unit_type;
    Unit::DisplayType display_type;
    Unit::CloakState cloak;
    int action;
    int tag;
    int owner;
    float energy;
    float energy_max;
    bool is_flying;
    bool is_burrowed;
    bool is_powered;
    float radius;
    float facing;
    float detect_range;
    float weapon_cooldown;
    float build_progress;
    float shield;
    float health;
    float health_max;
    Tag engaged_target_tag;

    SerializedUnit (const Unit* unit) {
        tag = unit->tag;
        position = SerializedPos { unit->pos.x, unit->pos.y };
        unit_type = unit->unit_type;
        canonical_unit_type = unit_type; // canonicalize(unit_type);
        display_type = unit->display_type;
        cloak = unit->cloak;
        owner = unit->owner;
        energy = unit->energy;
        energy_max = unit->energy_max;
        is_flying = unit->is_flying;
        is_burrowed = unit->is_burrowed;
        is_powered = unit->is_powered;
        radius = unit->radius;
        facing = unit->facing;
        detect_range = unit->detect_range;
        weapon_cooldown = unit->weapon_cooldown;
        build_progress = unit->build_progress;
        shield = unit->shield;
        health = unit->health;
        health_max = unit->health_max;
        engaged_target_tag = unit->engaged_target_tag;
    }

    template<class Archive>
    void serialize(Archive & archive) {
        archive(
            CEREAL_NVP(position),
            CEREAL_NVP(tag),
            CEREAL_NVP(unit_type),
            CEREAL_NVP(display_type),
            CEREAL_NVP(cloak),
            CEREAL_NVP(energy),
            CEREAL_NVP(energy_max),
            CEREAL_NVP(is_flying),
            CEREAL_NVP(is_burrowed),
            CEREAL_NVP(is_powered),
            CEREAL_NVP(radius),
            CEREAL_NVP(facing),
            CEREAL_NVP(detect_range),
            CEREAL_NVP(weapon_cooldown),
            CEREAL_NVP(build_progress),
            CEREAL_NVP(shield),
            CEREAL_NVP(health),
            CEREAL_NVP(health_max),
            CEREAL_NVP(engaged_target_tag),
        );
    }
};

struct State {
    int tick;
    int playerID;
    vector<SerializedUnit> units;
    vector<bool> walkableMap;

    template<class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(tick), CEREAL_NVP(playerID), CEREAL_NVP(units), CEREAL_NVP(walkableMap));
    }
};

struct Reward {
    float reward;
    int state1;
    int state2;
};

struct Session {
    vector<State> states;
    vector<float> rewards;
};

py::object predictFunction;
py::object addSession;

// Model
// Σ Pi + Σ PiSij
class MicroTrainer : public sc2::ReplayObserver {
    int tick = 0;
    vector<State> states;
    int fileIndex = 0;
    int opponentPlayerID;

   public:
    void OnGameLoading() {
    }

    void OnGameStart() override {
        states = vector<State>();
        Debug()->DebugEnemyControl();
        Debug()->DebugShowMap();
        Debug()->DebugIgnoreFood();
        Debug()->DebugIgnoreResourceCost();
        // Debug()->DebugGiveAllTech();
        initMappings(Observation());

        opponentPlayerID = Observation()->GetPlayerID() == 1 ? : 2 : 1;

        // DependencyAnalyzer deps;
        // deps.analyze(Observation());

        /*const sc2::UnitTypes& unit_types = Observation()->GetUnitTypeData();

        Point2D mn = Observation()->GetGameInfo().playable_min;
        Point2D mx = Observation()->GetGameInfo().playable_max;

        int i = 0;
        for (const UnitTypeData& type : unit_types) {
            if (!type.available)
                continue;
            if (i > 300) break;
            float x = mn.x + ((rand() % 1000) / 1000.0f) * (mx.x - mn.x);
            float y = mn.y + ((rand() % 1000) / 1000.0f) * (mx.y - mn.y);
            Debug()->DebugCreateUnit(type.unit_type_id, Point2D(x, y));
            i++;
        }
        

        Debug()->SendDebug();*/
    }

    void Reset() {
        Point2D mn = Observation()->GetGameInfo().playable_min;
        Point2D mx = Observation()->GetGameInfo().playable_max;

        for (int i = 0; i < 40; i++) {
            float x = mn.x + ((rand() % 1000) / 1000.0f) * (mx.x - mn.x);
            float y = mn.y + ((rand() % 1000) / 1000.0f) * (mx.y - mn.y);
            Debug()->DebugCreateUnit(UNIT_TYPEID::PROTOSS_ZEALOT, Point2D(x, y), opponentPlayerID);
        }

        Debug()->DebugCreateUnit(UNIT_TYPEID::TERRAN_REAPER, Point2D((mn.x + mx.x)*0.5f, (mn.y + mx.y)*0.5f), Observation()->GetPlayerID());
    }

    void OnGameEnd() override {
        fileIndex++;
        auto t0 = Clock::now();
        stringstream ss;
        ss << "output/data";
        ss << fileIndex;
        ss << ".json";
        ofstream os(ss.str());
        cereal::JSONOutputArchive archive(os);
        archive(states);
        auto t2 = Clock::now();
        cout << "Time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t0).count() << " ns" << endl;
    }

    void OnStep() override {
        State state;
        state.playerID = Observation()->GetPlayerID();
        state.tick = Observation()->GetGameLoop();
        auto ourUnits = Observation()->GetUnits(Unit::Alliance::Self);
        auto enemyUnits = Observation()->GetUnits(Unit::Alliance::Enemy);
        for (auto u : ourUnits) state.units.push_back(SerializedUnit(u));
        for (auto u : enemyUnits) state.units.push_back(SerializedUnit(u));
        
        if ((tick % 25) == 0) {
            cout << "Gathering state" << endl;
            states.push_back(state);
        }

        if (ourUnits.size() == 0) {
            Reset();
        }

        stringstream os;
        {
            cereal::JSONOutputArchive archive(os);
            archive(state);
        }
        Action action = (Action)predictFunction(os.str(), ourUnits[0]->tag).cast<int>();

        // Actions()->SendActions();
        // Debug()->SendDebug();
        tick++;
    }
};

int main(int argc, char* argv[]) {
    py::scoped_interpreter guard{};
    py::exec(R"(
        import sys
        sys.path.append("bot/micro_train")
    )");
    py::module trainer = py::module::import("micro_train");
    predictFunction = trainer.attr("predict");
    addSession = trainer.attr("addSession");

    /*py::print("Hello, World!");
    
        import micro_train
        print('Hello World!!!')
    )");*/


    /*Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetMultithreaded(true);

    MicroTrainer bot;
    agent = bot;
    coordinator.SetParticipants({ CreateParticipant(Race::Terran, &bot) });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    bot.OnGameLoading();
    coordinator.StartGame(EmptyMap);

    while (coordinator.Update() && !do_break) {
        if (PollKeyPress()) {
            do_break = true;
        }
    }
    return 0;*/

    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    // if (!coordinator.SetReplayPath(kReplayFolder)) {
    //     std::cout << "Unable to find replays." << std::endl;
    //     return 1;
    // }
    if (!coordinator.LoadReplayList(kReplayListProtoss)) {
        std::cout << "Unable to find replays." << std::endl;
        return 1;
    }

    MicroTrainer replay_observer1;

    coordinator.AddReplayObserver(&replay_observer1);

    while (coordinator.Update())
        ;
    while (!sc2::PollKeyPress())
        ;
}
