#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"
#include "bot_examples.h"
#include "Bot.h"

using namespace sc2;
using namespace BOT;

int main(int argc, char* argv[]) {
    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetMultithreaded(true);
    BOT::Bot bot;

    coordinator.SetParticipants({
        CreateParticipant(Race::Terran, &bot),
        CreateComputer(Race::Terran, Difficulty::VeryEasy)
    });

    // Start the game.
    coordinator.LaunchStarcraft();
    bool do_break = false;

    for (; !do_break;) {
        coordinator.StartGame(kMapBelShirVestigeLE);

        while (coordinator.Update() && !do_break) {
            if (PollKeyPress()) {
                do_break = true;
            }
        }
    }
    return 0;
}
