#include "Bot.h"
#include "bot_examples.h"
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"
#include "CompositionAnalyzer.h"
#include <iostream>
#include <string>

using namespace sc2;
using namespace BOT;

const char* BelShirVestigeLE = "Ladder/(2)Bel'ShirVestigeLE (Void).SC2Map";
const char* BackwaterLE = "Ladder/BackwaterLE.SC2Map";
const char* BlackpinkLE = "Ladder/BlackpinkLE.SC2Map";
const char* CatalystLE = "Ladder/CatalystLE.SC2Map";
const char* EastwatchLE = "Ladder/EastwatchLE.SC2Map";
const char* NeonVioletSquareLE = "Ladder/NeonVioletSquareLE.SC2Map";



int main(int argc, char* argv[]) {/*
    std::cout << argc << " " << (std::string(argv[1]) == "--composition") << std::endl;
    if (argc >= 2 && std::string(argv[1]) == "--composition") {
        RunCompositionAnalyzer(argc-1, argv);
        return 0;
    }*/

    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    // sc2::FeatureLayerSettings settings(kCameraWidth, kFeatureLayerSize, kFeatureLayerSize, kFeatureLayerSize, kFeatureLayerSize);
    // coordinator.SetFeatureLayers(settings);

    coordinator.SetMultithreaded(true);
   
    coordinator.SetParticipants({
        CreateParticipant(Race::Terran, &bot), CreateComputer(Race::Zerg, Difficulty::Hard),
    });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    for (; !do_break;) {
        bot.OnGameLoading();
        coordinator.StartGame(BelShirVestigeLE);

        while (coordinator.Update() && !do_break) {
            if (PollKeyPress()) {
                do_break = true;
            }
        }
    }

    return 0;
}
