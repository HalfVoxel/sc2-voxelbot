#include "CompositionAnalyzer.h"
#include <fstream>
#include <iostream>
#include <queue>
#include "behaviortree/MicroNodes.h"
#include "Bot.h"
#include "sc2api/sc2_api.h"
#include "sc2utils/sc2_manage_process.h"

using namespace sc2;
using namespace std;

static const int TIMEOUT_TICKS = 11 * 30;

vector<UNIT_TYPEID> unitTypes = {
    UNIT_TYPEID::TERRAN_LIBERATOR,
    UNIT_TYPEID::TERRAN_BATTLECRUISER,
    UNIT_TYPEID::TERRAN_BANSHEE,
    UNIT_TYPEID::TERRAN_CYCLONE,
    UNIT_TYPEID::TERRAN_GHOST,
    UNIT_TYPEID::TERRAN_HELLION,
    UNIT_TYPEID::TERRAN_HELLIONTANK,
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

vector<UNIT_TYPEID> opponentTypes = {
    UNIT_TYPEID::ZERG_BANELING,
    UNIT_TYPEID::ZERG_BROODLING,
    UNIT_TYPEID::ZERG_BROODLORD,
    UNIT_TYPEID::ZERG_CORRUPTOR,
    UNIT_TYPEID::ZERG_CREEPTUMORQUEEN,
    UNIT_TYPEID::ZERG_DRONE,
    UNIT_TYPEID::ZERG_HYDRALISK,
    UNIT_TYPEID::ZERG_INFESTOR,
    UNIT_TYPEID::ZERG_INFESTORTERRAN,
    UNIT_TYPEID::ZERG_LARVA,
    UNIT_TYPEID::ZERG_LOCUSTMP,
    UNIT_TYPEID::ZERG_LURKERMP,
    UNIT_TYPEID::ZERG_LURKERMPBURROWED,
    UNIT_TYPEID::ZERG_MUTALISK,
    UNIT_TYPEID::ZERG_OVERLORD,
    UNIT_TYPEID::ZERG_OVERSEER,
    UNIT_TYPEID::ZERG_QUEEN,
    UNIT_TYPEID::ZERG_RAVAGER,
    UNIT_TYPEID::ZERG_ROACH,
    UNIT_TYPEID::ZERG_SPINECRAWLER,
    UNIT_TYPEID::ZERG_SPORECRAWLER,
    UNIT_TYPEID::ZERG_SWARMHOSTBURROWEDMP,
    UNIT_TYPEID::ZERG_SWARMHOSTMP,
    UNIT_TYPEID::ZERG_ULTRALISK,
    UNIT_TYPEID::ZERG_VIPER,
    UNIT_TYPEID::ZERG_ZERGLING,
    UNIT_TYPEID::ZERG_PARASITICBOMBDUMMY,

    UNIT_TYPEID::PROTOSS_ADEPT,
    UNIT_TYPEID::PROTOSS_ARCHON,
    UNIT_TYPEID::PROTOSS_CARRIER,
    UNIT_TYPEID::PROTOSS_COLOSSUS,
    UNIT_TYPEID::PROTOSS_DARKTEMPLAR,
    UNIT_TYPEID::PROTOSS_DISRUPTOR,
    UNIT_TYPEID::PROTOSS_DISRUPTORPHASED,
    UNIT_TYPEID::PROTOSS_HIGHTEMPLAR,
    UNIT_TYPEID::PROTOSS_IMMORTAL,
    // UNIT_TYPEID::PROTOSS_INTERCEPTOR,
    UNIT_TYPEID::PROTOSS_MOTHERSHIP,
    // UNIT_TYPEID::PROTOSS_MOTHERSHIPCORE,
    UNIT_TYPEID::PROTOSS_OBSERVER,
    UNIT_TYPEID::PROTOSS_ORACLE,
    UNIT_TYPEID::PROTOSS_PHOENIX,
    // UNIT_TYPEID::PROTOSS_PHOTONCANNON,
    UNIT_TYPEID::PROTOSS_PROBE,
    // UNIT_TYPEID::PROTOSS_PYLONOVERCHARGED,
    UNIT_TYPEID::PROTOSS_SENTRY,
    // UNIT_TYPEID::PROTOSS_SHIELDBATTERY,
    UNIT_TYPEID::PROTOSS_STALKER,
    UNIT_TYPEID::PROTOSS_TEMPEST,
    UNIT_TYPEID::PROTOSS_VOIDRAY,
    UNIT_TYPEID::PROTOSS_WARPPRISM,
    UNIT_TYPEID::PROTOSS_WARPPRISMPHASING,
    UNIT_TYPEID::PROTOSS_ZEALOT,
};

// Model
// Σ Pi + Σ PiSij

bool CompositionAnalyzer::Site::IsDone() {
    return (simulator.localSimulations > 400 || simulator.que.size() == 0) && state == 0;
}

void CompositionAnalyzer::Site::Attack() {
    Point2D middle = (tileMn + tileMx) * 0.5;
    for (auto unit : units) {
        simulator.Actions()->UnitCommand(unit, ABILITY_ID::ATTACK, middle);
    }
}

void CompositionAnalyzer::Site::writeUnits(vector<pair<UNIT_TYPEID, int>> u) {
    // simulator.results << u.size() << "  ";
    for (auto p : u) {
        simulator.results << (int)p.first << " " << p.second << "  ";
    }
}

void CompositionAnalyzer::Site::writeResult() {
    vector<double> totalLife(2);
    vector<double> damageTaken(2);
    vector<double> weights(2);
    for (auto unit : units) {
        int id = unit->owner - simulator.Observation()->GetPlayerID();
        totalLife[id] += (unit->is_alive ? unit->health / (double)unit->health_max : 0.0);
        damageTaken[id] += (unit->is_alive ? (double)unit->health_max - unit->health : unit->health_max);
        weights[id]++;
    }

    if (weights[0] > 0)
        totalLife[0] /= weights[0];
    if (weights[1] > 0)
        totalLife[1] /= weights[1];

    Result result;
    for (auto v : queItem.first)
        result.side1.unitCounts.push_back({ v.first, v.second });
    for (auto v : queItem.second)
        result.side2.unitCounts.push_back({ v.first, v.second });
    result.side1.remainingLifeFraction = totalLife[0];
    result.side2.remainingLifeFraction = totalLife[1];
    result.side1.damageTaken = damageTaken[0];
    result.side2.damageTaken = damageTaken[1];

    stringstream json;
    {
        cereal::JSONOutputArchive archive(json);
        result.serialize(archive);
    }
    // writeUnits(queItem.first);
    // simulator.results << " ";
    // writeUnits(queItem.second);
    // simulator.results << totalLife[0] << " " << totalLife[1] << " " << damageTaken[0] << " " << damageTaken[1];
    // simulator.results << endl;
    simulator.results << json.str() << endl
                      << "------" << endl;
}

void CompositionAnalyzer::Site::kill() {
    auto mn = tileMn + (tileMx - tileMn) * 0.1;
    auto mx = tileMx - (tileMx - tileMn) * 0.1;
    for (auto unit : units) {
        simulator.Debug()->DebugKillUnit(unit);
    }
    units.clear();
    for (auto unit : simulator.Observation()->GetUnits()) {
        if (unit->pos.x >= mn.x && unit->pos.x < mx.x && unit->pos.y >= mn.y && unit->pos.y < mx.y) {
            simulator.Debug()->DebugKillUnit(unit);
        }
    }
}

void CompositionAnalyzer::Site::OnStep() {
    if (state == 0) {
        if (IsDone())
            return;

        notInCombat = 0;
        auto p = queItem = simulator.que.front();
        simulator.que.pop();
        simulator.simulation++;
        simulator.localSimulations++;

        cout << "Simulation " << simulator.simulation << " (" << simulator.que.size() << " remaining)" << endl;
        for (auto type : p.first)
            cout << type.second << " " << UnitTypeToName(type.first) << ", ";
        cout << endl;
        for (auto type : p.second)
            cout << type.second << " " << UnitTypeToName(type.first) << ", ";
        cout << endl;

        auto p1 = tileMn + (tileMx - tileMn) * 0.4;
        auto p2 = tileMn + (tileMx - tileMn) * 0.6;
        for (auto type : p.first)
            simulator.Debug()->DebugCreateUnit(type.first, p1, simulator.Observation()->GetPlayerID(), type.second);
        for (auto type : p.second)
            simulator.Debug()->DebugCreateUnit(type.first, p2, simulator.Observation()->GetPlayerID() + 1, type.second);

        state++;
    } else if (state == 1) {
        state++;
    } else if (state == 2) {
        auto mn = tileMn + (tileMx - tileMn) * 0.1;
        auto mx = tileMx - (tileMx - tileMn) * 0.1;
        units.clear();
        for (auto unit : simulator.Observation()->GetUnits()) {
            if (unit->pos.x >= mn.x && unit->pos.x < mx.x && unit->pos.y >= mn.y && unit->pos.y < mx.y) {
                units.push_back(unit);
            }
        }
        state++;
        Attack();
    } else if (state == 3) {
        if ((simulator.tick % 100) == 0) {
            Attack();
        }

        // Point2D middle = (tileMn + tileMx) * 0.5;
        vector<int> unitCounts(2);
        bool inCombat = false;
        for (auto unit : units) {
            if (unit->is_alive) {
                unitCounts[unit->owner - simulator.Observation()->GetPlayerID()]++;
                // simulator.Debug()->DebugLineOut(unit->pos, Point3D(middle.x, middle.y, 10));
                // simulator.Debug()->DebugTextOut(UnitTypeToName(unit->unit_type), unit->pos);
                if (unit->engaged_target_tag != NullTag) {
                    auto* target = simulator.Observation()->GetUnit(unit->engaged_target_tag);
                    if (target != nullptr && target->is_alive) {
                        inCombat = true;
                    }
                }
            }
        }

        if (!inCombat) {
            notInCombat++;
            // simulator.Debug()->DebugTextOut("Not In Combat", Point3D(middle.x, middle.y, 0));
        } else {
            notInCombat = 0;
            // simulator.Debug()->DebugTextOut("Combat", Point3D(middle.x, middle.y, 0));
        }

        if (notInCombat > TIMEOUT_TICKS) {
            cout << "TIMEOUT" << endl;

            writeResult();

            kill();

            state = 0;
        } else if (unitCounts[0] == 0 || unitCounts[1] == 0) {
            writeResult();

            // int winner = unitCounts[0] == 0 ? 1 : 0;
            kill();

            state = 0;
        }
    }
}

void CompositionAnalyzer::OnGameLoading() {
}

CompositionAnalyzer::CompositionAnalyzer() {
    results = ofstream("out.txt");
    for (int i = 0; i < unitTypes.size(); i++) {
        for (int j = 0; j < opponentTypes.size(); j++) {
            for (int k = 1; k < 10; k += 2) {
                for (int m = 1; m < 10; m += 2) {
                    vector<pair<UNIT_TYPEID, int>> left = { { unitTypes[i], k } };
                    vector<pair<UNIT_TYPEID, int>> right = { { opponentTypes[j], m } };
                    que.push({ left, right });
                }
            }
        }
    }

    for (int i = 0; i < 100; i++) {
    }
}

bool CompositionAnalyzer::ShouldReload() {
    for (auto site : sites) {
        if (!site.IsDone())
            return false;
    }
    return true;
}

void CompositionAnalyzer::OnGameStart() {
    localSimulations = 0;
    Debug()->DebugEnemyControl();
    Debug()->DebugShowMap();
    Debug()->DebugIgnoreFood();
    Debug()->DebugIgnoreResourceCost();
    Debug()->DebugGiveAllTech();

    sites.clear();
    int tiles = 5;
    Point2D mn = Observation()->GetGameInfo().playable_min;
    Point2D mx = Observation()->GetGameInfo().playable_max;
    double dx = (mx.x - mn.x) / tiles;
    double dy = (mx.y - mn.y) / tiles;
    for (int x = 0; x < tiles; x++) {
        for (int y = 0; y < tiles; y++) {
            auto tileMn = mn + Point2D(x * dx, y * dy);
            auto tileMx = mn + Point2D((x + 1) * dx, (y + 1) * dy);
            sites.push_back(Site(*this, tileMn, tileMx));
        }
    }
}

void CompositionAnalyzer::OnStep() {
    tick++;
    for (auto& site : sites)
        site.OnStep();

    TickMicro();

    Actions()->SendActions();
    Debug()->SendDebug();
}
