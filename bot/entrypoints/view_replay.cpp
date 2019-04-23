#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_api.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "../build_order_train/serialized_state.h"
#include "../ml/replay.h"
#include "../BuildOptimizerGenetic.h"
#include <random>
#include <iostream>
#include <fstream>
#include "../DependencyAnalyzer.h"
#include "../utilities/sc2_serialization.h"
#include "sc2utils/sc2_manage_process.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>

// Protoss:
// e5242d0a121db241ccfca68150feea57deeb82b9d7000e7d00c84b5cba4e511e.SC2Replay
using namespace std;
using namespace sc2;

const char* kReplayFolder = "/home/arong/learning/sc2-voxelbot/replays";
const char* kReplayListProtoss = "/home/arong/learning/sc2-voxelbot/pvp.txt";




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

    virtual bool IgnoreReplay(const ReplayInfo& replay_info, uint32_t& player_id) override {
        bool version_match = replay_info.base_build == Control()->Proto().GetBaseBuild() && replay_info.data_version == Control()->Proto().GetDataVersion();
        if (!version_match) {
            cerr << "Skipping replay because of version mismatch " << replay_info.base_build << " != " << Control()->Proto().GetBaseBuild() << " || " << replay_info.data_version << " != " << Control()->Proto().GetDataVersion() << endl;
        }

        if (replay_info.num_players != 2 || !version_match) {
            return true;
        }

        return false;
    }

    void OnGameStart() final {
    }

    void OnUnitCreated(const sc2::Unit* unit) final {
        return;
    }

    void OnStep() final {
    }
    
    void OnGameEnd() final {
    }
};

int main(int argc, char* argv[]) {

    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    // coordinator.SetStepSize(112); // 5 seconds (22.4*5)

    // if (!coordinator.SetReplayPath(kReplayFolder)) {
    //     std::cout << "Unable to find replays." << std::endl;
    //     return 1;
    // }
    vector<string> list = { "/Users/arong/Programming/kth/thesis/laddertest/replays/VoxelbotvKagamine-Bandwidth.SC2Replay"};
    coordinator.LoadReplayList(list);

    Replay replay_observer1;
    replay_observer1.playerID = 1;
    coordinator.AddReplayObserver(&replay_observer1);

    default_random_engine rnd(time(0));

    // coordinator.SetPortStart(mod.attr("getPort")().cast<int>());

    while (true) {
        bool done = !coordinator.Update();
        if (done) break;
    }
    cout << "Done" << endl;
    // while (!sc2::PollKeyPress());
}