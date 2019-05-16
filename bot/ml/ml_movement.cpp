#include "ml_movement.h"
#include "replay.h"
#include <libvoxelbot/utilities/mappings.h>
#include <libvoxelbot/utilities/predicates.h>
#include <libvoxelbot/utilities/python_utils.h>
#include <sstream>
#include <iostream>
#include "../Bot.h"

using namespace std;
using namespace sc2;

Race GetPlayerRace(const ObservationInterface* obs, int playerID) {
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
    return;
	pybind11::module modMovement = pybind11::module::import("predictor_movement");
    pybind11::module modMovementTarget = pybind11::module::import("predictor_movement_target");
	stepper = modMovement.attr("Stepper")();
    stepperTarget = modMovementTarget.attr("Stepper")();

	cout << "Loading weights" << endl;
	modMovement.attr("load_weights")("models/movement_10.weights");
    modMovementTarget.attr("load_weights")("models/movement_target_16.weights");
}

bool isMovableUnit(UNIT_TYPEID type) {
    return isBasicHarvester(type) || isArmy(type) || type == UNIT_TYPEID::ZERG_OVERLORD;
}

vector<const Unit*> lastMovedUnits;
vector<float> lastMoveProbs;
map<int, float> timeBias;

void MLMovement::Tick(const ObservationInterface* observation) {
    int playerID = observation->GetPlayerID();
    auto ourUnits = observation->GetUnits(Unit::Alliance::Self);
    #if false

    if ((ticks % 50) == 1) {
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

        // pybind11::gil_scoped_acquire acquire;
        lock_guard<mutex> lock(python_thread_mutex);

        string sessionString = json.str();
        auto res = stepper.attr("step")(sessionString, playerID).cast<vector<float>>();

        vector<const Unit*> validUnits;
        for (const Unit*& u : ourUnits) {
            if (u->owner == playerID && isMovableUnit(u->unit_type)) {
                validUnits.push_back(u);
            }
        }

        for (size_t i = 0; i < res.size(); i++) {
            timeBias[(int)validUnits[i]->tag] *= 0.9f;
            timeBias[(int)validUnits[i]->tag] += res[i] * 0.1f;
            res[i] += 2 * timeBias[(int)validUnits[i]->tag];
        }

        lastMovedUnits.clear();
        lastMoveProbs.clear();
        assert(validUnits.size() == res.size());

        vector<bool> isSampled;
        for (size_t i = 0; i < res.size(); i++) {
            if (res[i] > 0.05) {
                lastMovedUnits.push_back(validUnits[i]);
                lastMoveProbs.push_back(res[i]);
                isSampled.push_back(false);
            }
        }

        for (int k = 0; k < 2; k++) {
            // Reservoir sampling
            int armyFilter = 1;
            float totalWeight = 0;
            int sampledUnit = -1;
            for (size_t i = 0; i < res.size(); i++) {
                bool valid = (armyFilter == -1 || isArmy(validUnits[i]->unit_type) == armyFilter);
                if (valid && !isSampled[i]) {
                    if (((rand() % 10000)/10000.0f) * totalWeight <= res[i]) {
                        totalWeight += res[i];
                        sampledUnit = i;
                        // if (armyFilter == -1) armyFilter = isArmy(validUnits[i]->unit_type);
                    }
                }
            }

            vector<int> sampledTags;
            vector<const Unit*> sampledUnits;

            if (sampledUnit != -1) {
                // Pick other nearby units
                for (size_t j = 0; j < res.size(); j++) {
                    bool valid = (armyFilter == -1 || isArmy(validUnits[j]->unit_type) == armyFilter);
                    if (valid && !isSampled[j] && Distance2D(validUnits[j]->pos, validUnits[sampledUnit]->pos) < 8 && res[j] > res[sampledUnit]*0.5f) {
                        sampledTags.push_back((int)validUnits[j]->tag);
                        sampledUnits.push_back(validUnits[j]);
                        isSampled[j] = true;
                        timeBias[(int)validUnits[j]->tag] = 0;
                    }
                }
            }

            if (sampledTags.size() > 0) {
                vector<float> targetCoord;
                bool shouldKeepOrder;
                bool isAttackOrder;
                tie(targetCoord, shouldKeepOrder, isAttackOrder) = stepperTarget.attr("step")(sessionString, sampledTags, playerID).cast<tuple<vector<float>, bool, bool>>();

                if (!shouldKeepOrder) {
                    Point2D coord = Point2D(targetCoord[0], targetCoord[1]);

                    for (auto* u : sampledUnits) {
                        agent->Actions()->UnitCommand(u, isAttackOrder ? ABILITY_ID::ATTACK : ABILITY_ID::MOVE, coord);
                        agent->Debug()->DebugLineOut(u->pos, Point3D(coord.x, coord.y, u->pos.z));
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < lastMovedUnits.size(); i++) {
        agent->Debug()->DebugSphereOut(lastMovedUnits[i]->pos, min(lastMoveProbs[i], 0.5f) * 3, lastMoveProbs[i] > 0.5 ? Colors::Yellow : (lastMoveProbs[i] > 0.2 ? Colors::Red : Colors::Blue));
    }

    for (const Unit*& u : ourUnits) {
        if (u->owner == playerID && isMovableUnit(u->unit_type)) {
            if (u->orders.size() > 0) {
                auto& order = u->orders[0];
                if (Point2D(0.0f, 0.0f) != order.target_pos) {
                    agent->Debug()->DebugLineOut(u->pos, Point3D(order.target_pos.x, order.target_pos.y, u->pos.z), Colors::Red);
                }

                if (u->orders.size() > 1) {
                    cout << "Multiple orders ";
                    for (auto o : u->orders) {
                        cout << AbilityTypeToName(o.ability_id) << " ";
                        if (o.target_unit_tag != NullTag) cout << "(unit) ";
                        if (Point2D(0,0) != o.target_pos) cout << "(point) ";
                    }
                    cout << endl;
                }
            }
        }
    }
    #endif
}