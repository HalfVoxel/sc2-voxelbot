#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_api.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "../build_order_train/serialized_state.h"
#include "../ml/replay.h"
#include <libvoxelbot/buildorder/optimizer.h>
#include <random>
#include <iostream>
#include <fstream>
#include <libvoxelbot/utilities/sc2_serialization.h>
#include "sc2utils/sc2_manage_process.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>

// Protoss:
// e5242d0a121db241ccfca68150feea57deeb82b9d7000e7d00c84b5cba4e511e.SC2Replay
using namespace std;
using namespace sc2;

const char* kReplayFolder = "/home/arong/learning/sc2-voxelbot/replays";
const char* kReplayListProtoss = "/home/arong/learning/sc2-voxelbot/pvp.txt";
const string saveDir = "/home/arong/learning/sc2-voxelbot/training_data/replays/s3";




pybind11::object saveFunction;
pybind11::object isReplayAlreadySavedFunction;
pybind11::object replaySavePath;

class Replay : public sc2::ReplayObserver {
   public:
    std::vector<uint32_t> count_units_built_;

    vector<vector<int>> unit_implies_has_had_unit;
    vector<vector<int>> unit_implies_has_had_unit_total;

    ObserverSession session;
    int playerID;
    bool finished = false;

    Replay()
        : sc2::ReplayObserver() {
    }

    virtual int GetReplayPerspective () override {
        return playerID;
    }

    // virtual bool IgnoreReplay(const std::string& filepath) override {
    //     if (isReplayAlreadySavedFunction(replaySavePath(filepath, saveDir)).cast<bool>()) {
    //         cerr << "Skipping already processed replay" << endl;
    //         return true;
    //     }
    //     return false;
    // }

    virtual bool IgnoreReplay(const ReplayInfo& replay_info, uint32_t& /* player_id */) override {
        bool version_match = replay_info.base_build == Control()->Proto().GetBaseBuild() && replay_info.data_version == Control()->Proto().GetDataVersion();
        if (!version_match) {
            cerr << "Skipping replay because of version mismatch " << replay_info.base_build << " != " << Control()->Proto().GetBaseBuild() << " || " << replay_info.data_version << " != " << Control()->Proto().GetDataVersion() << endl;
        }

        if (replay_info.num_players != 2 || !version_match) {
            return true;
        }

        ReplayPlayerInfo p1;
        replay_info.GetPlayerInfo(p1, 1);

        ReplayPlayerInfo p2;
        replay_info.GetPlayerInfo(p2, 2);

        if (p1.race != Race::Protoss || p2.race != Race::Protoss) {
            cerr << "Skipping replay because matchup is not PvP (was " << RaceToString(p1.race) << "v" << RaceToString(p2.race) << ")" << endl;
            return true;
        }

        if (p1.mmr < 3500 && p2.mmr < 3500) {
            cerr << "Skipping replay because MMR is too low (" << p1.mmr << ", " << p2.mmr << ")" << endl;
            return true;
        }

        return false;
    }

    void OnGameStart() final {
        cout << "Started game..." << endl;
        // DependencyAnalyzer deps;
        // deps.analyze();
        // exit(0);
        stepIndex = 0;
        finished = false;
        session = ObserverSession();

        /*const sc2::ObservationInterface* obs = Observation();
        assert(obs->GetUnitTypeData().size() > 0);
        int numUnits = obs->GetUnitTypeData().size();
        count_units_built_ = vector<uint32_t>(numUnits, 0);
        if (unit_implies_has_had_unit.size() == 0) {
            unit_implies_has_had_unit = vector<vector<int>>(numUnits, vector<int>(numUnits, 0));
            unit_implies_has_had_unit_total = vector<vector<int>>(numUnits, vector<int>(numUnits, 0));
        }*/
        cout << "Playing " << playerID << ": " << RaceToString(GetPlayerInfo(playerID).race) << endl;
        cout << "against " << (3-playerID) << ": " << RaceToString(GetPlayerInfo(3 - playerID).race) << endl;

        collectState();
    }

    void OnUnitCreated(const sc2::Unit* unit) final {
        return;
        assert(uint32_t(unit->unit_type) < count_units_built_.size());
        ++count_units_built_[unit->unit_type];

        auto ourUnits = Observation()->GetUnits(Unit::Alliance::Self);
        // auto enemyUnits = Observation()->GetUnits(Unit::Alliance::Enemy);
        for (int i = 0; i < count_units_built_.size(); i++) {
            if (count_units_built_[i] > 0) {
                unit_implies_has_had_unit[unit->unit_type][i]++;
            }
            unit_implies_has_had_unit_total[unit->unit_type][i]++;
        }
    }

    int stepIndex = 0;
    void OnStep() final {
        assert((int)Observation()->GetPlayerID() == playerID);

        cout << "Step " << Observation()->GetGameLoop() << endl;
        // About every 10 seconds (faster game speed)
        collectState();

        stepIndex += 1;
    }

    const ReplayPlayerInfo GetPlayerInfo(int playerID) {
        ReplayPlayerInfo info;
        if (!ReplayControl()->GetReplayInfo().GetPlayerInfo(info, playerID)) {
            throw std::invalid_argument("playerID out of range " + std::to_string(playerID));
        }
        return info;
    }

    void collectState() {
        auto units = Observation()->GetUnits(Unit::Alliance::Self);
        int playerID = Observation()->GetPlayerID();
        int opponentPlayer = 3 - playerID;
        assert(playerID == 1 || playerID == 2);

        float time = ticksToSeconds(Observation()->GetGameLoop());
        BuildState state(Observation(), Unit::Alliance::Self, GetPlayerInfo(playerID).race, BuildResources(Observation()->GetMinerals(), Observation()->GetVespene()), time);
        SerializedState serializedState(state);
        for (auto u : Observation()->GetUpgrades()) {
            serializedState.upgrades.push_back(u);
        }
        BuildState enemyState(Observation(), Unit::Alliance::Enemy, GetPlayerInfo(opponentPlayer).race, BuildResources(0, 0), time);
        session.observations.selfStates.push_back(serializedState);
        session.observations.enemyStates.push_back(SerializedState(enemyState));
        session.observations.rawUnits.push_back(Observation()->GetUnits());
    }
    
    void OnGameEnd() final {
        finished = true;
        // std::cout << "Units created:" << std::endl;
        const sc2::ObservationInterface* obs = Observation();

        session.gameInfo = obs->GetGameInfo();

        session.winner = -1;
        for (auto r : obs->GetResults()) {
            if (r.result == GameResult::Win) {
                session.winner = r.player_id;
            }
            cout << r.player_id << " is a " << (r.result == GameResult::Win ? "win" : (r.result == GameResult::Loss ? "loss" : "tie/?")) << endl;
            if (r.result == GameResult::Tie) {
                session.winner = -2;
                cerr << "Tie" << endl;
            }
        }

        session.replayInfo = ReplayControl()->GetReplayInfo();
        
        
        /*const sc2::UnitTypes& unit_types = obs->GetUnitTypeData();
        for (uint32_t i = 0; i < count_units_built_.size(); ++i) {
            if (count_units_built_[i] == 0) {
                continue;
            }

            std::cout << unit_types[i].name << ": " << std::to_string(count_units_built_[i]) << std::endl;
        }

        for (uint32_t i = 0; i < count_units_built_.size(); ++i) {
            for (uint32_t j = 0; j < count_units_built_.size(); ++j) {
                if (unit_implies_has_had_unit[i][j] == unit_implies_has_had_unit_total[i][j] && unit_implies_has_had_unit_total[i][j] > 2) {
                    cout << unit_types[i].name << " implies " << unit_types[j].name << endl;
                } else if (unit_implies_has_had_unit[i][j] > unit_implies_has_had_unit_total[i][j] * 0.9f) {
                    cout << unit_types[i].name << " softly implies " << unit_types[j].name << " (" << (unit_implies_has_had_unit[i][j] / (double)unit_implies_has_had_unit_total[i][j]) << endl;
                }
            }
        }*/

        /*int numUnits = unit_implies_has_had_unit.size();
        unit_direct_implications = vector<vector<int>> (numUnits, vector<int>(numUnits, 0));
        for (uint32_t i = 0; i < count_units_built_.size(); ++i) {
            for (uint32_t j = 0; j < count_units_built_.size(); ++j) {
                if (unit_implies_has_had_unit_total[i][j] <= 2) continue;

                float implies = unit_implies_has_had_unit[i][j] / (float)unit_implies_has_had_unit_total[i][j];
                for (uint32_t i2 = 0; i2 < count_units_built_.size(); ++i2) {
                    if (unit_implies_has_had_unit_total[i][i2] <= 2) continue;

            }
        }*/
    }
};

void saveSession (ReplaySession& session) {
    stringstream json;
    {
        cereal::JSONOutputArchive archive(json);
        session.serialize(archive);
    }

    saveFunction(json.str(), replaySavePath(session.replayInfo.replay_path, saveDir));
}

int main(int argc, char* argv[]) {
    initMappings();

    // printMappings();

    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        sys.path.append("bot/python")
    )");
    pybind11::module mod = pybind11::module::import("replay_saver");
    
    saveFunction = mod.attr("save");
    isReplayAlreadySavedFunction = mod.attr("isReplayAlreadySaved");
    replaySavePath = mod.attr("replaySavePath");
    vector<string> replays = mod.attr("findReplays")(kReplayListProtoss).cast<vector<string>>();


    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetStepSize(112); // 5 seconds (22.4*5)

    // if (!coordinator.SetReplayPath(kReplayFolder)) {
    //     std::cout << "Unable to find replays." << std::endl;
    //     return 1;
    // }
    coordinator.LoadReplayList(replays);

    Replay replay_observer1;
    replay_observer1.playerID = 1;
    coordinator.AddReplayObserver(&replay_observer1);

    Replay replay_observer2;
    replay_observer2.playerID = 2;
    coordinator.AddReplayObserver(&replay_observer2);

    coordinator.SetPortStart(mod.attr("getPort")().cast<int>());

    while (true) {
        bool done = !coordinator.Update();
        if (replay_observer1.finished) {
            assert(replay_observer2.finished);
            auto session = ReplaySession(replay_observer1.session, replay_observer2.session);
            saveSession(session);
            cout << "Saved session" << endl;
        }

        if (done) break;
    }
    cout << "Done" << endl;
    // while (!sc2::PollKeyPress());
}