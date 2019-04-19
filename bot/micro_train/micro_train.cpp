#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include "../Bot.h"
#include "../DependencyAnalyzer.h"
#include "../utilities/mappings.h"
#include "../generated/abilities.h"
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include "../utilities/stdutils.h"
#include "cereal/cereal.hpp"

#include <thread>

const char* kReplayFolder = "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays";
const char* kReplayListProtoss = "/Users/arong/Programming/kth/multi-agent/MultiAgentSystemsA4/replays_protoss.txt";

using Clock = std::chrono::high_resolution_clock;

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

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(x), CEREAL_NVP(y));
    }
};

enum Action {
    Flee,
    MoveRandom,
    Attack_Closest,
    Idle,
    MoveToAlly,
    MoveAwayFromAlly,
};

vector<string> actionName = {
    "Flee",
    "Move Random",
    "Attack Closest",
    "Idle",
    "Move to Ally",
    "Move away from Ally",
};

vector<Point2D> action2dir = {
    Point2D(0, 1),    // N
    Point2D(-1, 1),   // NW
    Point2D(-1, 0),   // W
    Point2D(-1, -1),  // SW
    Point2D(0, -1),   // S
    Point2D(1, -1),   // SE
    Point2D(1, 0),    // E
    Point2D(1, 1),    // NE
    Point2D(0, 1),    // Attack N
    Point2D(-1, 0),   // Attack W
    Point2D(0, -1),   // Attack S
    Point2D(1, 0),    // Attack E
};

struct SerializedUnit {
    SerializedPos position;
    UNIT_TYPEID unit_type;
    UNIT_TYPEID canonical_unit_type;
    Unit::DisplayType display_type;
    Unit::CloakState cloak;
    int action;
    Tag tag;
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
    float shield_max;
    float health;
    float health_max;
    Tag engaged_target_tag;
    int last_attacked_tick;

    SerializedUnit(const SerializedUnit* lastUnit, const Unit* unit, int tick) {
        tag = unit->tag;
        position = SerializedPos{ unit->pos.x, unit->pos.y };
        unit_type = unit->unit_type;
        canonical_unit_type = unit_type;  // canonicalize(unit_type);
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
        shield_max = unit->shield_max;
        health = unit->health;
        health_max = unit->health_max;
        engaged_target_tag = unit->engaged_target_tag;

        if (lastUnit != nullptr) {
            if (health + shield < lastUnit->health + lastUnit->shield) {
                last_attacked_tick = tick;
            } else {
                last_attacked_tick = lastUnit->last_attacked_tick;
            }
        } else {
            last_attacked_tick = -1000;
        }
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(
            CEREAL_NVP(position),
            CEREAL_NVP(tag),
            CEREAL_NVP(unit_type),
            CEREAL_NVP(display_type),
            CEREAL_NVP(cloak),
            CEREAL_NVP(owner),
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
            CEREAL_NVP(shield_max),
            CEREAL_NVP(engaged_target_tag),
            CEREAL_NVP(action),
            CEREAL_NVP(last_attacked_tick));
    }
};

struct State {
    int tick;
    int playerID;
    vector<SerializedUnit> units;
    vector<bool> walkableMap;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(tick), CEREAL_NVP(playerID), CEREAL_NVP(units), CEREAL_NVP(walkableMap));
    }
};

struct Session {
    vector<State> states;
    int ticks = 0;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(states));
    }
};

const SerializedUnit* try_get(std::map<Tag, const SerializedUnit*>& m, Tag key) {
    return m.find(key) != m.end() ? m[key] : nullptr;
}

py::object predictFunction;
py::object rewardFunction;
py::object addSession;
py::object optimizeFunction;

struct UnitDebug {
    vector<string> messages;
};

class MicroTrainer : public sc2::Agent {
    int tick = 0;
    Session session;
    int fileIndex = 0;
    int opponentPlayerID;
    bool simulateRealtime = false;
    bool paused = false;
    bool enableExploration = true;
    bool interactive = false;
    State state;
    map<Tag, UnitDebug*> unitDebug;
    vector<float> lastActionQValues;
    vector<float> selfTensor;
    vector<vector<float>> allyTensor;
    vector<vector<float>> enemyTensor;

   public:
    int resets = 1;

    void OnGameLoading() {
    }

    bool ShouldReload() {
        return (resets % 30) == 0;
    }

    void OnGameStart() override {
        cout << "Starting..." << endl;
        session = Session();
        state = State();

        Debug()->DebugEnemyControl();
        Debug()->DebugShowMap();
        Debug()->DebugIgnoreFood();
        Debug()->DebugIgnoreResourceCost();
        // Debug()->DebugGiveAllTech();
        initMappings(Observation());

        opponentPlayerID = Observation()->GetPlayerID() == 1 ? 2 : 1;

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
        cout << "Started..." << endl;
    }

    void CompleteSession() {
        if (session.states.size() <= 1) {
            session = Session();
            return;
        }
        cout << "Completed session" << endl;
        fileIndex++;
        auto t0 = Clock::now();
        stringstream json;
        {
            cereal::JSONOutputArchive archive(json);
            session.serialize(archive);
        }
        addSession(json.str());
        optimizeFunction(session.states.size());
        session = Session();
        state = State();

        auto t2 = Clock::now();
        cout << "Time: " << chrono::duration_cast<chrono::nanoseconds>(t2 - t0).count() << " ns" << endl;
    }

    void Reset() {
        cout << "Reset "
             << "(" << resets << ")" << endl;
        CompleteSession();

        resets++;
        if (ShouldReload()) {
            return;
        }

        for (auto unit : Observation()->GetUnits()) {
            Debug()->DebugKillUnit(unit);
        }

        // Initial
        Point2D mn = Observation()->GetGameInfo().playable_min;
        Point2D mx = Observation()->GetGameInfo().playable_max;

        // for (int i = 0; i < 100; i++) {
        //     float x = mn.x + ((rand() % 1000) / 1000.0f) * (mx.x - mn.x);
        //     float y = mn.y + ((rand() % 1000) / 1000.0f) * (mx.y - mn.y);
        //     Debug()->DebugCreateUnit(UNIT_TYPEID::PROTOSS_ZEALOT, Point2D(x, y), opponentPlayerID);
        // }

        for (int i = 0; i < 40; i++) {
            float x = mn.x + ((rand() % 1000) / 1000.0f) * (mx.x - mn.x);
            float y = mn.y + ((rand() % 1000) / 1000.0f) * (mx.y - mn.y);

            int num = 1;  // (rand() % 2) + 1;
            for (int j = 0; j < num; j++) {
                Debug()->DebugCreateUnit(UNIT_TYPEID::TERRAN_MARINE, Point2D(x, y), opponentPlayerID);
            }
        }

        Debug()->DebugCreateUnit(UNIT_TYPEID::TERRAN_REAPER, Point2D((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f), Observation()->GetPlayerID());
        Debug()->DebugCreateUnit(UNIT_TYPEID::TERRAN_REAPER, Point2D((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f), Observation()->GetPlayerID());
        Debug()->DebugCreateUnit(UNIT_TYPEID::TERRAN_REAPER, Point2D((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f), Observation()->GetPlayerID());
        // Debug()->DebugCreateUnit(UNIT_TYPEID::TERRAN_REAPER, Point2D((mn.x + mx.x)*0.5f, (mn.y + mx.y)*0.5f), Observation()->GetPlayerID());
    }

    void OnGameEnd() override {
    }

    void DoAction(const Unit* unit, Action action) {
        /*Point2D moveDir = action2dir[action];
        if (action == Action::Attack_N || action == Action::Attack_W || action == Action::Attack_S || action == Action::Attack_E) {
            Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, unit->pos + moveDir*5);
        } else {
            Actions()->UnitCommand(unit, ABILITY_ID::MOVE, unit->pos + moveDir*5);
        }*/
        auto enemyUnits = Observation()->GetUnits(Unit::Alliance::Enemy);
        auto allyUnits = Observation()->GetUnits(Unit::Alliance::Self);
        const Unit* closestUnit = nullptr;
        float closestDist = 10000000;
        Point2D avgEnemyPos;
        float totalWeight = 0;
        for (auto enemy : enemyUnits) {
            auto d = DistanceSquared2D(unit->pos, enemy->pos);
            if (d < closestDist) {
                closestDist = d;
                closestUnit = enemy;
            }

            float w = max(0.0f, 1.0f - sqrt(d) / 10.0f);
            if (w > 0) {
                totalWeight += w;
                avgEnemyPos += w * enemy->pos;
            }
        }

        if (totalWeight > 0) {
            avgEnemyPos /= totalWeight;
        } else {
            avgEnemyPos = unit->pos;
        }

        Point2D avgAllyPos;
        double totalAllyWeight = 0;
        for (auto u : allyUnits) {
            if (u != unit) {
                auto d = Distance2D(unit->pos, u->pos);

                double w = exp(-d / 5.0);
                if (w > 0.0000001) {
                    totalAllyWeight += w;
                    avgAllyPos += w * u->pos;
                }
            }
        }

        if (totalAllyWeight > 0) {
            avgAllyPos /= totalAllyWeight;
        } else {
            avgAllyPos = unit->pos;
        }

        closestDist = DistanceSquared2D(avgEnemyPos, unit->pos);
        Debug()->DebugSphereOut(Point3D(avgEnemyPos.x, avgEnemyPos.y, unit->pos.z), 0.5);
        Debug()->DebugSphereOut(Point3D(avgAllyPos.x, avgAllyPos.y, unit->pos.z), 0.2);

        /*if (unit->weapon_cooldown == 0) {
            if (closestDist > 0 && closestDist < 2*2) action = Action::Flee;
            else action = Action::Attack_Closest;
        } else {
            if (closestDist > 0 && closestDist < 5*5) action = Action::Flee;
            else action = Action::Attack_Closest;
        }

        cout << "Action: " << action << " " << closestDist << " " << (closestUnit != nullptr) << " " << unit->weapon_cooldown << endl;*/

        if ((action == Action::MoveToAlly || action == Action::MoveAwayFromAlly) && totalAllyWeight == 0) {
            action = Action::MoveRandom;
        }

        switch (action) {
            case Action::Attack_Closest: {
                if (closestUnit != nullptr) {
                    Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, closestUnit->pos);
                } else {
                    Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, unit->pos);
                }
                break;
            }
            case Action::MoveRandom: {
                float x = tick / 250.0f + unit->pos.x / 100;
                float dx = -0.143 * sin(1.75 * (x + 1.73)) - 0.18 * sin(2.96 * (x + 4.98)) - 0.012 * sin(6.23 * (x + 3.17)) + 0.088 * sin(8.07 * (x + 4.63));
                float y = (tick + 512357) / 250.0f + unit->pos.y / 100;
                float dy = -0.143 * sin(1.75 * (y + 1.73)) - 0.18 * sin(2.96 * (y + 4.98)) - 0.012 * sin(6.23 * (y + 3.17)) + 0.088 * sin(8.07 * (y + 4.63));

                // float dx = ((rand() % 10000) / 5000.0f) - 1.0f;
                // float dy = ((rand() % 10000) / 5000.0f) - 1.0f;
                Actions()->UnitCommand(unit, ABILITY_ID::MOVE, unit->pos + Point2D(dx, dy) * 10);
                break;
            }
            case Action::Flee: {
                if (closestUnit != nullptr) {
                    float dist = sqrt(closestDist);
                    Actions()->UnitCommand(unit, ABILITY_ID::MOVE, unit->pos + ((unit->pos - avgEnemyPos) / (dist + 0.001)) * 10);
                } else {
                    Actions()->UnitCommand(unit, ABILITY_ID::MOVE, unit->pos);
                }
                break;
            }
            case Action::MoveAwayFromAlly: {
                float dist = Distance2D(unit->pos, avgAllyPos);
                Actions()->UnitCommand(unit, ABILITY_ID::MOVE, unit->pos + ((unit->pos - avgAllyPos) / (dist + 0.001)) * 10);
                break;
            }
            case Action::MoveToAlly: {
                Actions()->UnitCommand(unit, ABILITY_ID::MOVE, avgAllyPos);
                break;
            }
            case Idle: {
                Actions()->UnitCommand(unit, ABILITY_ID::MOVE, unit->pos);
                break;
            }
        }
    }

    void OnStep() override {
        auto ourUnits = Observation()->GetUnits(Unit::Alliance::Self);

        for (auto message : Observation()->GetChatMessages()) {
            if (message.message == "r" || message.message == "realtime") {
                simulateRealtime = !simulateRealtime;
                Actions()->SendChat(simulateRealtime ? "Enabled realtime mode" : "Disabled realtime mode");
            }
            if (message.message == "p" || message.message == "pause") {
                paused = !paused;
                Actions()->SendChat(paused ? "Paused" : "Unpaused");
            }
            if (message.message == "e" || message.message == "exploration") {
                enableExploration = !enableExploration;
                Actions()->SendChat(enableExploration ? "Enabled exploration" : "Disabled exploration");
            }
            if (message.message == "i" || message.message == "interactive") {
                interactive = !interactive;
                Actions()->SendChat(interactive ? "Enabled interactive mode" : "Disabled interactive mode");
            }

            if (message.message.size() == 1 && message.message[0] > '0' && message.message[0] <= '9') {
                Action action = (Action)(message.message[0] - '0');
                DoAction(ourUnits[0], action);
                stringstream ss;
                ss << "Took action " << actionName[action];
                Actions()->SendChat(ss.str());
            }
        }

        if (simulateRealtime) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        if (paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }

        session.ticks++;

        if (ourUnits.size() > 0) {
            Point2D avgPos = Point2D(0, 0);
            float totalWeight = 0;
            for (auto unit : ourUnits) {
                avgPos += unit->pos;
                totalWeight += 1;
            }
            avgPos /= totalWeight;

            Debug()->DebugMoveCamera(avgPos);
        }

        State newState;
        newState.playerID = Observation()->GetPlayerID();
        newState.tick = tick;
        auto enemyUnits = Observation()->GetUnits(Unit::Alliance::Enemy);
        map<Tag, const SerializedUnit*> lastUnitsMap;
        for (const SerializedUnit& u : state.units)
            lastUnitsMap[u.tag] = &u;

        for (auto u : ourUnits)
            newState.units.push_back(SerializedUnit(try_get(lastUnitsMap, u->tag), u, tick));
        for (auto u : enemyUnits)
            newState.units.push_back(SerializedUnit(try_get(lastUnitsMap, u->tag), u, tick));
        state = move(newState);

        if (interactive) {
        } else if ((tick % 10) == 0) {
            stringstream os;
            {
                cereal::JSONOutputArchive archive(os);
                state.serialize(archive);
            }

            vector<Tag> unitTags;
            for (auto unit : ourUnits) {
                unitTags.push_back(unit->tag);
            }

            vector<tuple<int, vector<float>, vector<float>, vector<vector<float>>, vector<vector<float>>>> actions = predictFunction(os.str(), unitTags, enableExploration).cast<vector<tuple<int, vector<float>, vector<float>, vector<vector<float>>, vector<vector<float>>>>>();
            vector<float> rewards = rewardFunction(os.str(), unitTags).cast<vector<float>>();
            for (int i = 0; i < ourUnits.size(); i++) {
                UnitDebug*& debug = unitDebug[ourUnits[i]->tag];
                if (debug == nullptr)
                    debug = new UnitDebug();

                Action action = (Action)get<0>(actions[i]);
                lastActionQValues = get<1>(actions[i]);
                selfTensor = get<2>(actions[i]);
                allyTensor = get<3>(actions[i]);
                enemyTensor = get<4>(actions[i]);

                DoAction(ourUnits[i], action);
                // Note: they come in exactly the same order in the state
                state.units[i].action = action;

                stringstream ss;
                ss << "Took action " << actionName[action];
                // Actions()->SendChat(ss.str());

                stringstream ss2;
                ss2 << "Reward: " << rewards[i] << " " << actionName[action] << " ";
                debug->messages.push_back(ss2.str());

                stringstream ss3;
                for (int j = max(0, (int)debug->messages.size() - 5); j < debug->messages.size(); j++) {
                    ss3 << debug->messages[j] << endl;
                }
                //Debug()->DebugTextOut(ss3.str(), ourUnits[i]->pos);
                Debug()->DebugTextOut(ss3.str(), Point2D(0.6, 0.6), Colors::White, 8);
            }

            cout << "Gathering state" << endl;
            session.states.push_back(state);

            if (ourUnits.size() == 0 || session.ticks > 25 * 60 * 5) {
                Reset();
            }

            stringstream ss4;
            for (int i = 0; i < lastActionQValues.size(); i++) {
                ss4 << actionName[i] << ": ";
                for (int j = actionName[i].size(); j <= 20; j++)
                    ss4 << " ";
                ss4 << (int)lastActionQValues[i] << endl;
            }
            ss4 << endl
                << "Self Tensor" << endl;
            for (auto v : selfTensor)
                ss4 << setprecision(2) << v << endl;
            ss4 << endl
                << "Ally Tensor" << endl;
            for (auto ally : allyTensor) {
                for (auto v : ally)
                    ss4 << setprecision(2) << v << endl;
                ss4 << endl;
            }
            ss4 << endl
                << "Enemy Tensor" << endl;
            for (auto enemy : enemyTensor) {
                for (auto v : enemy)
                    ss4 << setprecision(2) << v << endl;
                ss4 << endl;
            }

            Debug()->DebugTextOut(ss4.str(), Point2D(0.6, 0.2), Colors::White, 8);
        }

        Actions()->SendActions();
        Debug()->SendDebug();
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
    rewardFunction = trainer.attr("get_reward");
    addSession = trainer.attr("addSession");
    optimizeFunction = trainer.attr("optimize");

    /*py::print("Hello, World!");
    
        import micro_train
        print('Hello World!!!')
    )");*/

    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetMultithreaded(true);

    MicroTrainer bot;
    agent = &bot;
    coordinator.SetParticipants({ CreateParticipant(Race::Terran, &bot) });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    bot.OnGameLoading();
    coordinator.StartGame(EmptyMap);

    while (coordinator.Update() && !do_break) {
        if (bot.ShouldReload()) {
            bot.resets++;
            coordinator.StartGame(EmptyMap);
        }
        // if (PollKeyPress()) {
        //     do_break = true;
        // }
    }
    return 0;

    /*sc2::Coordinator coordinator;
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
        ;*/
}
