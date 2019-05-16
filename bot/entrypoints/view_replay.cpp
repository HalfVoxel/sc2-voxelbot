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
#include "sc2utils/sc2_arg_parser.h"

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

struct ReplayOptions {
	string path = "";
	int perspective = 1;
};

static void ParseArguments(int argc, char *argv[], ReplayOptions& options) {
	sc2::ArgParser arg_parser(argv[0]);

	std::vector<sc2::Arg> args = {
		sc2::Arg("-r", "--replay", "Replay path", true),
		sc2::Arg("-p", "--perspective", "Perspective (0 = both, 1 = player1, 2 = player2", false),
	};
	arg_parser.AddOptions(args);

	arg_parser.Parse(argc, argv);
	std::string p;
	if (arg_parser.Get("replay", p)) {
		options.path = p;
	}
    if (arg_parser.Get("perspective", p)) {
        stringstream ss(p);
        ss >> options.perspective;
	}
}

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
    // vector<string> list = { "/Users/arong/Programming/kth/thesis/laddertest/replays/VoxelbotvAdditionalPylons-Bandwidth.SC2Replay"};

    ReplayOptions options;
    ParseArguments(argc, argv, options);
    vector<string> list = { options.path };
    coordinator.LoadReplayList(list);

    Replay replay_observer1;
    replay_observer1.playerID = options.perspective;
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