#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"
#include <iostream>
#include <queue>

using namespace sc2;
using namespace std;

const char* EmptyMap = "Test/Empty.SC2Map";

vector<UNIT_TYPEID> unitTypes = {
    UNIT_TYPEID::TERRAN_BANSHEE,
    UNIT_TYPEID::TERRAN_BATTLECRUISER,
    UNIT_TYPEID::TERRAN_CYCLONE,
    UNIT_TYPEID::TERRAN_GHOST,
    UNIT_TYPEID::TERRAN_HELLION,
    UNIT_TYPEID::TERRAN_HELLIONTANK,
    UNIT_TYPEID::TERRAN_LIBERATOR,
    UNIT_TYPEID::TERRAN_LIBERATORAG,
    UNIT_TYPEID::TERRAN_MARAUDER,
    UNIT_TYPEID::TERRAN_MARINE,
    UNIT_TYPEID::TERRAN_MEDIVAC,
    UNIT_TYPEID::TERRAN_MISSILETURRET,
    UNIT_TYPEID::TERRAN_MULE,
    UNIT_TYPEID::TERRAN_RAVEN,
    UNIT_TYPEID::TERRAN_REAPER,
    UNIT_TYPEID::TERRAN_SCV,
    UNIT_TYPEID::TERRAN_SIEGETANK,
    UNIT_TYPEID::TERRAN_SIEGETANKSIEGED,
    UNIT_TYPEID::TERRAN_THOR,
    UNIT_TYPEID::TERRAN_THORAP,
    UNIT_TYPEID::TERRAN_VIKINGASSAULT,
    UNIT_TYPEID::TERRAN_VIKINGFIGHTER,
    UNIT_TYPEID::TERRAN_WIDOWMINE,
    UNIT_TYPEID::TERRAN_WIDOWMINEBURROWED,
};

class CompositionAnalyzer : public sc2::Agent {
public:
    void OnGameLoading() {

    }

    void OnGameStart() override {
        Debug()->DebugEnemyControl();
        Debug()->DebugShowMap();
        Debug()->DebugIgnoreFood();
        Debug()->DebugIgnoreResourceCost();

        for (int i = 0; i < unitTypes.size(); i++) {
            for (int j = 0; j < unitTypes.size(); j++) {
                que.push(make_pair(unitTypes[i], unitTypes[j]));
            }
        }
    }

    queue<pair<UNIT_TYPEID,UNIT_TYPEID>> que;
    int index = 0;
    int state = 0;

    const int TIMEOUT_TICKS = 11 * 30;
    int notInCombat = 0;
    int tick = 0;
    void OnStep() override {
        tick++;
        // cout << Observation()->GetUnits().size() << endl;
        vector<Point2D> points = { {50,50}, {70, 50} };
        Point2D middle = (points[0] + points[1]) * 0.5;
        if (state == 0) {
            notInCombat = 0;
            auto p = que.front();
            que.pop();
            Point2D p1 (50,50);
            Point2D p2 (70,50);
            Debug()->DebugCreateUnit(p.first, p1, Observation()->GetPlayerID(), 1);
            Debug()->DebugCreateUnit(p.second, p2, Observation()->GetPlayerID()+1, 10);
            
            state++;
        } else if (state == 1) {
            state++;
        } else if (state == 2) {
            if ((tick % 100) == 0) {
                for (auto unit : Observation()->GetUnits()) {
                    Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, middle);
                }
            }

            vector<int> unitCounts(2);
            bool inCombat = false;
            for (auto unit : Observation()->GetUnits()) {
                unitCounts[unit->owner - Observation()->GetPlayerID()]++;
                if (unit->engaged_target_tag != NullTag) {
                    inCombat = true;
                }
            }

            if (!inCombat) {
                notInCombat++;
            } else {
                notInCombat = 0;
            }

            if (notInCombat > TIMEOUT_TICKS) {
                cout << "TIMEOUT" << endl;
                for (auto unit : Observation()->GetUnits()) {
                    Debug()->DebugKillUnit(unit);
                }

                state = 0;
            } else if (unitCounts[0] == 0 || unitCounts[1] == 0) {
                int winner = unitCounts[0] == 0 ? 1 : 0;
                for (auto unit : Observation()->GetUnits()) {
                    Debug()->DebugKillUnit(unit);
                }

                state = 0;
            }
        }

        Actions()->SendActions();
        Debug()->SendDebug();
    }
};

void RunCompositionAnalyzer(int argc, char* argv[]) {
    Coordinator coordinator;
    if (!coordinator.LoadSettings(argc, argv)) {
        return;
    }

    coordinator.SetMultithreaded(true);

    CompositionAnalyzer bot;
    coordinator.SetParticipants({
        CreateParticipant(Race::Terran, &bot)
    });

    // Start the game.

    coordinator.SetRealtime(false);
    coordinator.LaunchStarcraft();
    bool do_break = false;

    for (; !do_break;) {
        bot.OnGameLoading();
        coordinator.StartGame(EmptyMap);

        while (coordinator.Update() && !do_break) {
            if (PollKeyPress()) {
                do_break = true;
            }
        }
    }
}
