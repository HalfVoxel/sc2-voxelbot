#include "Bot.h"
#include "UnitNodes.h"
#include "bot_examples.h"
#include "sc2api/sc2_api.h"
#include "sc2api/sc2_typeenums.h"
#include "sc2utils/sc2_manage_process.h"

using namespace sc2;
using namespace BOT;

const char* BelShirVestigeLE = "Ladder/(2)Bel'ShirVestigeLE (Void).SC2Map";
const char* BackwaterLE = "Ladder/BackwaterLE.SC2Map";
const char* BlackpinkLE = "Ladder/BlackpinkLE.SC2Map";
const char* CatalystLE = "Ladder/CatalystLE.SC2Map";
const char* EastwatchLE = "Ladder/EastwatchLE.SC2Map";
const char* NeonVioletSquareLE = "Ladder/NeonVioletSquareLE.SC2Map";



int main(int argc, char* argv[]) {
    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetMultithreaded(true);
   
    coordinator.SetParticipants({
        CreateParticipant(Race::Terran, &bot), CreateComputer(Race::Zerg, Difficulty::Medium),
    });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    for (; !do_break;) {
        coordinator.StartGame(EastwatchLE);

        while (coordinator.Update() && !do_break) {
            if (PollKeyPress()) {
                do_break = true;
            }
        }
    }

    return 0;
}
