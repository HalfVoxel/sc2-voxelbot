#include "ml_movement.h"
#include "replay.h"
#include <sstream>
#include <iostream>

using namespace std;
using namespace sc2;

const Race GetPlayerRace(const ObservationInterface* obs, int playerID) {
	return obs->GetGameInfo().player_info[playerID].race_actual;
}

ObserverSession collectState(const ObservationInterface* obs) {
	auto units = obs->GetUnits(Unit::Alliance::Self);
    int playerID = obs->GetPlayerID();
    int opponentPlayer = 3 - playerID;
    assert(playerID == 1 || playerID == 2);

    float time = ticksToSeconds(obs->GetGameLoop());
    BuildState state(obs, Unit::Alliance::Self, GetPlayerRace(obs, playerID), BuildResources(obs->GetMinerals(), obs->GetVespene()), time);
    SerializedState serializedState(state);
    for (auto u : obs->GetUpgrades()) {
        serializedState.upgrades.push_back(u);
    }
    BuildState enemyState(obs, Unit::Alliance::Enemy, GetPlayerRace(obs, opponentPlayer), BuildResources(0, 0), time);

    ObserverSession session;
    session.observations.selfStates.push_back(serializedState);
    session.observations.enemyStates.push_back(SerializedState(enemyState));
    session.observations.rawUnits.push_back(obs->GetUnits());
    session.gameInfo = obs->GetGameInfo();
    session.replayInfo = ReplayInfo();
    session.replayInfo.duration = time;
    session.replayInfo.duration_gameloops = obs->GetGameLoop();
    session.replayInfo.map_name = session.gameInfo.map_name;
    session.replayInfo.map_path = session.gameInfo.local_map_path;
    session.replayInfo.replay_path = "invalid";
    session.winner = -1;
    return session;
}


void MLMovement::OnGameStart() {
	pybind11::module modMovement = pybind11::module::import("predictor_movement");
	stepper = modMovement.attr("Stepper")();

	cout << "Loading weights" << endl;
	modMovement.attr("load_weights")("models/movement_41.weights");
}


void MLMovement::Tick(const ObservationInterface* observation) {
	// Create state
	// Serialize state
	// Predict
	// Convert to unit orders
	auto session = collectState(observation);
	stringstream json;
    {
        cereal::JSONOutputArchive archive(json);
        session.serialize(archive);
    }

    stepper.attr("step")(json.str(), observation->GetPlayerID()).cast<vector<float>>();
}