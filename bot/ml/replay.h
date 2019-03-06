#pragma once
#include "sc2api/sc2_api.h"
#include "sc2api/sc2_interfaces.h"
#include "sc2api/sc2_map_info.h"
#include "../build_order_train/serialized_state.h"
#include "../utilities/sc2_serialization.h"
#include <vector>
#include <string>

struct RawState {
    std::vector<sc2::Unit> units;

    RawState() {}

    RawState(std::vector<const sc2::Unit*> units) {
        for (auto u : units) {
            this->units.push_back(*u);
        }
    }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(units));
    }
};

struct PlayerObservations {
    std::vector<SerializedState> selfStates;
    std::vector<SerializedState> enemyStates;
    std::vector<RawState> rawUnits;

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(selfStates), CEREAL_NVP(enemyStates), CEREAL_NVP(rawUnits));
    }
};

struct ObserverSession {
    PlayerObservations observations;
    int winner = -1;
    sc2::GameInfo gameInfo;
    sc2::ReplayInfo replayInfo;

    template <class Archive>
    inline void serialize(Archive& archive) {
        archive(CEREAL_NVP(observations), CEREAL_NVP(winner), CEREAL_NVP(gameInfo), CEREAL_NVP(replayInfo));
    }
};



struct ReplaySession {
    std::vector<PlayerObservations> observations = std::vector<PlayerObservations>(2);
    int winner = -1;
    sc2::GameInfo gameInfo;
    sc2::ReplayInfo replayInfo;
    std::vector<int> mmrs;

    ReplaySession() {}
    ReplaySession(const ObserverSession& player1, const ObserverSession& player2);

    template <class Archive>
    inline void serialize(Archive& archive) {
        archive(CEREAL_NVP(observations), CEREAL_NVP(winner), CEREAL_NVP(gameInfo), CEREAL_NVP(replayInfo), CEREAL_NVP(mmrs));
    }
};

std::string RaceToString(sc2::Race race);
