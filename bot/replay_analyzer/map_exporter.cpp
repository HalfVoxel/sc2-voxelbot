#include "sc2api/sc2_agent.h"
#include "sc2api/sc2_api.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "../build_order_train/serialized_state.h"
#include <libvoxelbot/utilities/influence.h>
#include <random>
#include <iostream>
#include <fstream>
#include "sc2utils/sc2_manage_process.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>

const char* kReplayFolder = "/home/arong/learning/sc2-voxelbot/replays";
const char* kReplayListProtoss = "/home/arong/learning/sc2-voxelbot/map_replays.txt";
// Protoss:
// e5242d0a121db241ccfca68150feea57deeb82b9d7000e7d00c84b5cba4e511e.SC2Replay
using namespace std;
using namespace sc2;

pybind11::object saveFunction;
pybind11::object fileExistsFunction;

class Replay : public sc2::ReplayObserver {
   public:

    Replay()
        : sc2::ReplayObserver() {
    }

    virtual int GetReplayPerspective () override {
        return 1;
    }

    virtual bool IgnoreReplay(const ReplayInfo& replay_info, uint32_t& player_id) override {
        bool version_match = replay_info.base_build == Control()->Proto().GetBaseBuild() && replay_info.data_version == Control()->Proto().GetDataVersion();
        if (!version_match) {
            cerr << "Skipping replay because of version mismatch " << replay_info.base_build << " != " << Control()->Proto().GetBaseBuild() << " || " << replay_info.data_version << " != " << Control()->Proto().GetDataVersion() << endl;
        }

        if (replay_info.num_players != 2 || !version_match) {
            return true;
        }

        auto map_name = replay_info.map_name;

        stringstream ss;
        ss << "training_data/maps/1/" << map_name << ".pickle";

        if (fileExistsFunction(ss.str()).cast<bool>()) {
            cerr << "Skipping already seen map " << map_name << endl;
            return true;
        }
        cout << "Starting " << map_name << endl;

        return false;
    }

    void OnGameStart() final {
        cout << "Started game..." << endl;

        auto gameInfo = Observation()->GetGameInfo();
        cout << "Analyzing " << gameInfo.map_name << endl;
        // 1 at traversable cells, 0 at walls
        auto pathing_grid = InfluenceMap(gameInfo.pathing_grid);

        // Remove the original command center/nexus/hatchery from the pathfinding map
        // (it is marked as blocked at the start of the game, we don't want that)
        vector<vector<double>> square5x5 = { { 0, 1, 1, 1, 0 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 1, 1, 1, 1, 1 }, { 0, 1, 1, 1, 0 } };
        pathing_grid.addInfluence(square5x5, Observation()->GetStartLocation() - Point2D(0.5, 0.5));
        for (auto p : gameInfo.enemy_start_locations) {
            pathing_grid.addInfluence(square5x5, p - Point2D(0.5, 0.5));
        }

        stringstream ss;
        ss << "training_data/maps/1/" << gameInfo.map_name << ".pickle";

        vector<vector<double>> grid (pathing_grid.w, vector<double>(pathing_grid.h));
        for (int x = 0; x < pathing_grid.w; x++) {
            for (int y = 0; y < pathing_grid.h; y++) {
                grid[x][y] = pathing_grid(x,y);
            }
        }

        auto tup = pybind11::list();
        tup.append(gameInfo.map_name);
        tup.append(gameInfo.local_map_path);
        tup.append(grid);
        saveFunction(tup, ss.str());
    }

    void OnStep() final {
    }
    
    void OnGameEnd() final {
    }
};



int main(int argc, char* argv[]) {
    initMappings();

    // printMappings();

    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        sys.path.append("bot/python")
    )");
    pybind11::module mod = pybind11::module::import("replay_saver");
    
    saveFunction = mod.attr("saveTensors");
    fileExistsFunction = mod.attr("isReplayAlreadySaved");


    sc2::Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetStepSize(112); // 5 seconds (22.4*5)

    vector<string> replays = mod.attr("findReplays")(kReplayListProtoss).cast<vector<string>>();
    coordinator.LoadReplayList(replays);

    Replay replay_observer1;
    coordinator.AddReplayObserver(&replay_observer1);
    coordinator.SetPortStart(10000);

    default_random_engine rnd(time(0));

    while (coordinator.Update()) {
    }
    cout << "Done" << endl;
    // while (!sc2::PollKeyPress());
}