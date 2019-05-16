#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <string>
#include "../CompositionAnalyzer.h"
#include "bot_examples.h"
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"
#include "../utilities/ladder_interface.h"
#include "sc2utils/sc2_arg_parser.h"

using namespace sc2;
using namespace BOT;
using namespace std;

const char* BelShirVestigeLE = "Ladder/(2)Bel'ShirVestigeLE (Void).SC2Map";
const char* BackwaterLE = "Ladder/BackwaterLE.SC2Map";
const char* BlackpinkLE = "Ladder/BlackpinkLE.SC2Map";
const char* CatalystLE = "Ladder/CatalystLE.SC2Map";
const char* EastwatchLE = "Ladder/EastwatchLE.SC2Map";
const char* NeonVioletSquareLE = "Ladder/NeonVioletSquareLE.SC2Map";
const char* ParaSiteLE = "Ladder/ParaSiteLE.SC2Map";

struct ConnectionOptionsLocal
{
	bool ComputerOpponent = false;
	sc2::Difficulty ComputerDifficulty = sc2::Difficulty::VeryEasy;
	sc2::Race ComputerRace = sc2::Race::Random;
	std::string ResultSavePath = "";
};

static void ParseArgumentsLocal(int argc, char *argv[], ConnectionOptionsLocal& connect_options)
{
	sc2::ArgParser arg_parser(argv[0]);

	std::vector<sc2::Arg> args = {
		sc2::Arg("-c", "--ComputerOpponent", "If we set up a computer oppenent", false),
		sc2::Arg("-a", "--ComputerRace", "Race of computer oppent", false),
		sc2::Arg("-d", "--ComputerDifficulty", "Difficulty of computer oppenent", false),
		sc2::Arg("-s", "--SavePath", "Path to resulting save path", false),
	};
	arg_parser.AddOptions(args);

	arg_parser.Parse(argc, argv);
	std::string sp;
	if (arg_parser.Get("SavePath", sp)) {
		connect_options.ResultSavePath = sp;
	}
	std::string CompOpp;
	if (arg_parser.Get("ComputerOpponent", CompOpp))
	{
        cout << "Got computer opponent" << endl;
		connect_options.ComputerOpponent = true;
		std::string CompRace;
		if (arg_parser.Get("ComputerRace", CompRace))
		{
			connect_options.ComputerRace = GetRaceFromString(CompRace);
		}
		std::string CompDiff;
		if (arg_parser.Get("ComputerDifficulty", CompDiff))
		{
			connect_options.ComputerDifficulty = GetDifficultyFromString(CompDiff);
		}
	}
}

static bool RunBotLocal(int argc, char *argv[], sc2::Agent *Agent,sc2::Race race)
{
	ConnectionOptionsLocal Options;
	ParseArgumentsLocal(argc, argv, Options);

	sc2::Coordinator coordinator;
	if (!coordinator.LoadSettings(argc, argv)) {
		return true;
	}

	// Add the custom bot, it will control the players.
	int num_agents;
	if (Options.ComputerOpponent) {
		num_agents = 1;
		coordinator.SetParticipants({
			CreateParticipant(race, Agent),
			CreateComputer(Options.ComputerRace, Options.ComputerDifficulty)
			});
	} else {
        assert(false);
	}
    ((Bot*)agent)->resultSavePath = Options.ResultSavePath;

	// Start the game.

	// Step forward the game simulation.
#if !DISABLE_PYTHON
    pybind11::module mod = pybind11::module::import("replay_saver");
    coordinator.SetPortStart(mod.attr("getPort")().cast<int>());
#endif
    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();

    bot->OnGameLoading();
    coordinator.StartGame(EastwatchLE);

    while (coordinator.Update()) {
        // if (PollKeyPress()) {
        //     do_break = true;
        // }
    }
    return true;
}


int main(int argc, char* argv[]) {
#if !DISABLE_PYTHON
    pybind11::scoped_interpreter guard{};
    pybind11::exec(R"(
        import sys
        import os
        sys.path.append("bot/python")
        os.environ["MPLBACKEND"] = "TkAgg"
    )");
#endif

    initMappings();
    bot = new Bot();
    agent = bot;

    if (RunBot(argc, argv, agent, Race::Protoss)) return 0;

    if (RunBotLocal(argc, argv, agent, Race::Protoss)) return 0;

    /*
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
    coordinator.SetPortStart(8020);

    coordinator.SetMultithreaded(true);
    // coordinator.SetUseGeneralizedAbilityId(false);

    coordinator.SetParticipants({
        CreateParticipant(Race::Protoss, bot),
        // CreateComputer(Race::Protoss, Difficulty::HardVeryHard),
        CreateComputer(Race::Terran, Difficulty::VeryHard),
    });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    for (; !do_break;) {
        bot->OnGameLoading();
        coordinator.StartGame(ParaSiteLE);

        while (coordinator.Update() && !do_break) {
            // if (PollKeyPress()) {
            //     do_break = true;
            // }
        }
    }

    return 0;
}
