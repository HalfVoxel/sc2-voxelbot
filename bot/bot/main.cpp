#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include "../CompositionAnalyzer.h"
#include "bot_examples.h"
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"

using namespace sc2;
using namespace BOT;


const char* BelShirVestigeLE = "Ladder/(2)Bel'ShirVestigeLE (Void).SC2Map";
const char* BackwaterLE = "Ladder/BackwaterLE.SC2Map";
const char* BlackpinkLE = "Ladder/BlackpinkLE.SC2Map";
const char* CatalystLE = "Ladder/CatalystLE.SC2Map";
const char* EastwatchLE = "Ladder/EastwatchLE.SC2Map";
const char* NeonVioletSquareLE = "Ladder/NeonVioletSquareLE.SC2Map";
const char* ParaSiteLE = "Ladder/ParaSiteLE.SC2Map";

int main(int argc, char* argv[]) { /*
    std::cout << argc << " " << (std::string(argv[1]) == "--composition") << std::endl;
    if (argc >= 2 && std::string(argv[1]) == "--composition") {
        RunCompositionAnalyzer(argc-1, argv);
        return 0;
    }*/

#if !DISABLE_PYTHON
    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        import os
        sys.path.append("bot/python")
        os.environ["MPLBACKEND"] = "TkAgg"
    )");
#endif

    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    // sc2::FeatureLayerSettings settings(kCameraWidth, kFeatureLayerSize, kFeatureLayerSize, kFeatureLayerSize, kFeatureLayerSize);
    // coordinator.SetFeatureLayers(settings);
    coordinator.SetPortStart(8020);

    coordinator.SetMultithreaded(true);

    coordinator.SetParticipants({
        CreateParticipant(Race::Protoss, &bot),
        CreateComputer(Race::Protoss, Difficulty::HardVeryHard),
    });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    for (; !do_break;) {
        bot.OnGameLoading();
        coordinator.StartGame(ParaSiteLE);

        while (coordinator.Update() && !do_break) {
            // if (PollKeyPress()) {
            //     do_break = true;
            // }
        }
    }

    return 0;
}
