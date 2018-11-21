#include <iostream>
#include <string>
#include "../CompositionAnalyzer.h"
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"

using namespace std;
using namespace sc2;

const char* EmptyMap = "Test/Empty.SC2Map";

int main(int argc, char* argv[]) {
    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetMultithreaded(true);

    CompositionAnalyzer bot;
    agent = bot;
    coordinator.SetParticipants({ CreateParticipant(Race::Terran, &bot) });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    bot.OnGameLoading();
    coordinator.StartGame(EmptyMap);

    while (coordinator.Update() && !do_break) {
        if (bot.ShouldReload()) {
            coordinator.StartGame(EmptyMap);
        }

        if (PollKeyPress()) {
            do_break = true;
        }
    }
    return 0;
}
