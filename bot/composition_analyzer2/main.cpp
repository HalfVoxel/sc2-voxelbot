#include <fstream>
#include <iostream>
#include <queue>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include "../Bot.h"
#include <libvoxelbot/combat/simulator.h>
#include "../CompositionAnalyzer.h"
#include "../DependencyAnalyzer.h"
#include <libvoxelbot/utilities/mappings.h>
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"

using namespace sc2;
using namespace std;

static const char* EmptyMap = "Test/units.SC2Map";

void printSide(const SideResult& side) {
    for (auto unitCount : side.unitCounts) {
        cout << "\t" << unitCount.count << "x " << UnitTypeToName(unitCount.type) << (isFlying(unitCount.type) ? " (flying)" : "") << endl;
    }
}

// Model
// Σ Pi + Σ PiSij
class CompositionAnalyzer2 : public sc2::Agent {
    CombatPredictor predictor;

   public:
    void OnGameLoading() {
    }

    void OnGameStart() override {
        initMappings();
        BuildOptimizerNN buildTimePredictor;
        buildTimePredictor.init();
        predictor.init();
        predictor.unitTest(buildTimePredictor);
        Debug()->DebugEnemyControl();
        Debug()->DebugShowMap();
        return;

        /*ifstream input("out_complete.txt");
        while(input) {
            stringstream ss;
            while(input) {
                string s;
                getline(input, s);
                if (s == "------") {
                    break;
                }
                if (s == "") {
                    continue;
                }
                ss << s << '\n';
            }
            if (ss.str().size() == 0) break;
            cereal::JSONInputArchive ar(ss);
            Result result;
            result.serialize(ar);

            if (abs(result.side1.remainingLifeFraction - result.side2.remainingLifeFraction) < 0.01) {
                // Ambigious
                continue;
            }

            CombatState state;
            int owner = 1;
            for (const SideResult& side : {result.side1, result.side2}) {
                for (auto unitCount : side.unitCounts) {
                    auto unit = CombatUnit(owner, unitCount.type, maxHealth(unitCount.type), isFlying(unitCount.type));
                    unit.shield = unit.shield_max = maxShield(unitCount.type);
                    for (int i = 0; i < unitCount.count; i++) {
                        state.units.push_back(unit);
                    }
                }
                owner++;
            }

            CombatState outcome = predictor.predict_engage(state).state;
            int winner = outcome.owner_with_best_outcome();

            int actualWinner = result.side1.remainingLifeFraction > result.side2.remainingLifeFraction ? 1 : 2;

            if (winner == actualWinner) {
                cout << "Prediction was accurate" << endl;
            } else {
                cout << "Prediction failed " << result.side1.remainingLifeFraction << " " << result.side2.remainingLifeFraction << " predicted " << winner << " but the actual winner was " << actualWinner << endl;
                cout << "Side 1" << endl;
                printSide(result.side1);
                cout << "Side 2" << endl;
                printSide(result.side2);
                for (auto u : state.units) {
                    cout << u.owner << " " << UnitTypeToName(u.type) << " " << u.health << " " << u.is_flying << endl;
                }
            }
        }*/
    }

    unique_ptr<CombatRecorder> recorder = nullptr;

    void OnStep() override {
        for (auto& message : Observation()->GetChatMessages()) {
            cout << "Read message '" << message.message << "'" << endl;
            if (message.message == "rec") {
                if (recorder != nullptr) {
                    Actions()->SendChat("Finished recording");
                    recorder->finalize();
                    recorder = nullptr;
                } else {
                    Actions()->SendChat("Starting recording");
                    recorder = make_unique<CombatRecorder>();
                }
            }
        }

        if (recorder != nullptr && (Observation()->GetGameLoop() % 10) == 0) {
            recorder->tick(Observation());
        }

        Actions()->SendActions();
        Debug()->SendDebug();
    }
};

int main(int argc, char* argv[]) {
    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        print(sys.path)
        sys.path.append("bot/python")
    )");

    if (argc == 2 && string(argv[1]) == "--simple") {
        initMappings();
        CombatPredictor predictor;
        BuildOptimizerNN buildTimePredictor;
        buildTimePredictor.init();
        predictor.init();
        predictor.unitTest(buildTimePredictor);
        exit(0);
    }
    

    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return 1;
    }

    coordinator.SetMultithreaded(true);

    CompositionAnalyzer2 bot;
    agent = &bot;
    coordinator.SetParticipants({ CreateParticipant(Race::Terran, &bot) });

    // Start the game.

    coordinator.SetRealtime(true);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    bot.OnGameLoading();
    coordinator.StartGame(EmptyMap);

    while (coordinator.Update() && !do_break) {
        // if (PollKeyPress()) {
        //     do_break = true;
        // }
    }
    return 0;
}
