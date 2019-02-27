#include "replay.h"

using namespace std;
using namespace sc2;

float combatStrength(SerializedState& state) {
    float count = 0;
    for (auto& u : state.units) {
        count += (isArmy(u.type) ? 1 : 0.1f) * getUnitData(u.type).food_required * u.totalCount;
    }
    return count;
}

string RaceToString(Race race) {
    switch(race) {
        case Race::Terran:
            return "Terran";
        case Race::Protoss:
            return "Protoss";
        case Race::Zerg:
            return "Zerg";
        default:
            return "Unknown";
    }
}

ReplaySession::ReplaySession(const ObserverSession& player1, const ObserverSession& player2) {
    observations = { player1.observations, player2.observations };
    assert(player1.winner == player2.winner);
    winner = player1.winner;
    gameInfo = player1.gameInfo;
    replayInfo = player1.replayInfo;
    assert(player1.replayInfo.duration_gameloops == player2.replayInfo.duration_gameloops);
    ReplayPlayerInfo player1info;
    ReplayPlayerInfo player2info;
    player1.replayInfo.GetPlayerInfo(player1info, 1);
    player2.replayInfo.GetPlayerInfo(player2info, 2);
    mmrs = { player1info.mmr, player2info.mmr };
    cout << "Player MMRs: " << player1info.mmr << " " << player2info.mmr << endl;

    if (winner != 1 && winner != 2) {
        cerr << "Unknown game result!" << endl;
    } else {
        cout << "Winner is " << winner << " " << RaceToString(observations[winner-1].selfStates[0].race) << endl;
        float a1 = combatStrength(observations[(winner-1)].selfStates.back());
        float a2 = combatStrength(observations[1-(winner-1)].selfStates.back());
        if (a1 > a2) {
            cout << "Winner seems consistent with game state" << endl;
        } else {
            cout << "Winner not consistent with game state " << a1 << " < " << a2 << endl;
        }
        // Save session

        
    }
}
